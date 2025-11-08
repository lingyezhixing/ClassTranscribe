# ClassTranscribe/app.py
import asyncio
import datetime
import json
import logging
import os
import re
import httpx
import yaml
import shutil
import difflib
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from AudioSplitter import AudioSplitter

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局状态变量 ---
app_state: Dict[str, Any] = {
    "config": None,
    "audio_splitter": None,
    "openai_client": None,
    "vad_queue": asyncio.Queue(),
    "asr_queue": asyncio.Queue(),
    "llm_queue": asyncio.Queue(),
    "background_tasks": [],
    "scanning_active": False,
    "vad_tasks": [],
    "asr_tasks": {},
    "llm_tasks": {}
}

# --- 辅助函数 (无变化) ---
def load_config(path: str = "config.yaml") -> Dict:
    logging.info(f"正在从 {path} 加载配置...")
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    for pair in config.get("path_pairs", []):
        os.makedirs(pair["scan_path"], exist_ok=True)
        os.makedirs(pair["transfer_path"], exist_ok=True)
    logging.info("配置加载成功。")
    return config

async def run_in_threadpool(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

def merge_llm_results(results: List[str]) -> str:
    if not results: return ""
    merged_text = results[0]
    for i in range(1, len(results)):
        next_text = results[i]
        len1, len2 = len(merged_text), len(next_text)
        overlap_len = min(len1, len2, 200) 
        seq_matcher = difflib.SequenceMatcher(None, merged_text[-overlap_len:], next_text[:overlap_len])
        match = seq_matcher.find_longest_match(0, overlap_len, 0, overlap_len)
        if match.size > 5:
            merged_text += next_text[match.b + match.size:]
        else:
             merged_text += "\n" + next_text
    return merged_text

# --- 后台任务 ---

async def vad_scan_loop():
    """定期扫描新文件，并将其加入VAD队列"""
    cfg = app_state["config"]
    interval = cfg["scan_interval_minutes"] * 60
    
    while True:
        logging.info("VAD Scanner: 开始新一轮扫描...")
        try:
            for pair in cfg["path_pairs"]:
                scan_path = pair["scan_path"]
                
                files_in_dir = await run_in_threadpool(
                    lambda: [os.path.join(scan_path, f) for f in os.listdir(scan_path) if os.path.isfile(os.path.join(scan_path, f))]
                )
                
                existing_files = {task['file'] for task in app_state["vad_tasks"]}

                for filepath in files_in_dir:
                    if filepath not in existing_files:
                        app_state["vad_tasks"].append({"file": filepath, "status": "排队中"})
                        await app_state["vad_queue"].put(filepath)
                        logging.info(f"VAD Scanner: 新文件已入队: {filepath}")

        except Exception as e:
            logging.error(f"VAD Scanner: 扫描循环出错: {e}")

        await asyncio.sleep(interval)

async def vad_worker():
    """从VAD队列中获取任务并进行处理"""
    cfg = app_state["config"]
    splitter = app_state["audio_splitter"]
    
    while True:
        original_filepath = await app_state["vad_queue"].get()
        
        current_task = next((task for task in app_state["vad_tasks"] if task["file"] == original_filepath), None)
        if not current_task:
            app_state["vad_queue"].task_done()
            continue
            
        logging.info(f"VAD Worker: 开始处理 {original_filepath}")
        current_task["status"] = "处理中"
        
        try:
            filename = os.path.basename(original_filepath)
            transfer_path = cfg["path_pairs"][0]["transfer_path"]
            
            base_name, ext = os.path.splitext(filename)
            date_pattern = re.compile(r"^\d{4}年\d{2}月\d{2}日$")
            base_name_A = base_name if date_pattern.match(base_name) else datetime.date.today().strftime("%Y年%m月%d日")

            all_files_in_transfer = await run_in_threadpool(
                lambda: [f for f in os.listdir(transfer_path) if os.path.isfile(os.path.join(transfer_path, f))]
            )
            related_audio_files = sorted([f for f in all_files_in_transfer if os.path.splitext(f)[0].split('-')[0] == base_name_A])
            num_existing = len(related_audio_files)

            if num_existing == 0:
                new_filename = f"{base_name_A}{ext}"
            
            elif num_existing == 1:
                existing_filename = related_audio_files[0]
                existing_base_name, existing_ext = os.path.splitext(existing_filename)

                if '-' not in existing_base_name:
                    # 重命名音频文件
                    old_file_path = os.path.join(transfer_path, existing_filename)
                    new_indexed_filename = f"{base_name_A}-1{existing_ext}"
                    new_file_path = os.path.join(transfer_path, new_indexed_filename)
                    await run_in_threadpool(os.rename, old_file_path, new_file_path)
                    logging.info(f"VAD Worker: 为第一份文件添加序号: {existing_filename} -> {new_indexed_filename}")

                    # <--- 修复点: 同步重命名对应的文件夹 ---
                    old_dir_path = os.path.join(transfer_path, existing_base_name)
                    new_indexed_dir_name = f"{base_name_A}-1"
                    new_dir_path = os.path.join(transfer_path, new_indexed_dir_name)

                    if await run_in_threadpool(os.path.isdir, old_dir_path):
                        await run_in_threadpool(os.rename, old_dir_path, new_dir_path)
                        logging.info(f"VAD Worker: 同步重命名文件夹: {existing_base_name} -> {new_indexed_dir_name}")
                    # --- 修复点结束 ---
                
                new_filename = f"{base_name_A}-2{ext}"

            else:
                new_filename = f"{base_name_A}-{num_existing + 1}{ext}"

            final_path = os.path.join(transfer_path, new_filename)
            await run_in_threadpool(shutil.move, original_filepath, final_path)
            
            output_dir = os.path.join(transfer_path, os.path.splitext(new_filename)[0])
            logging.info(f"VAD Worker: 开始切分 '{final_path}'")
            await run_in_threadpool(
                splitter.split,
                audio_path=final_path, output_dir=output_dir, target_length=13,
                max_length=15, overlap_length=1, normalize_audio=True,
                norm_processes=cfg["vad_norm_processes"]
            )
            logging.info(f"VAD Worker: 文件 '{final_path}' 切分完成。")
            current_task["status"] = "完成"

        except Exception as e:
            logging.error(f"VAD Worker: 处理文件 '{original_filepath}' 时出错: {e}")
            current_task["status"] = f"失败: {e}"
        
        finally:
            await asyncio.sleep(2)
            try:
                app_state["vad_tasks"].remove(current_task)
            except ValueError:
                pass
            app_state["vad_queue"].task_done()

async def transcription_scan_loop():
    cfg = app_state["config"]
    allow_concurrent = cfg.get("allow_concurrent_asr_llm", False)
    logging.info(f"转录扫描器模式: {'并发' if allow_concurrent else '顺序'}")
    
    while True:
        if app_state["scanning_active"]:
            try:
                for pair in cfg["path_pairs"]:
                    transfer_path = pair["transfer_path"]
                    items = await run_in_threadpool(lambda: os.listdir(transfer_path))
                    
                    for item in items:
                        item_path = os.path.join(transfer_path, item)
                        if await run_in_threadpool(os.path.isdir, item_path):
                            task_id = item_path
                            vad_complete_file = os.path.join(item_path, "_vad_complete.json")
                            asr_complete_file = os.path.join(item_path, "_asr_complete.json")

                            has_vad = await run_in_threadpool(os.path.exists, vad_complete_file)
                            has_asr = await run_in_threadpool(os.path.exists, asr_complete_file)

                            if allow_concurrent:
                                if has_vad and not has_asr and task_id not in app_state["asr_tasks"]:
                                    app_state["asr_tasks"][task_id] = "排队中"
                                    await app_state["asr_queue"].put(task_id)
                                    logging.info(f"扫描器 (并发): 新ASR任务已入队: {task_id}")
                                
                                if has_vad and has_asr and task_id not in app_state["llm_tasks"]:
                                    app_state["llm_tasks"][task_id] = "排队中"
                                    await app_state["llm_queue"].put(task_id)
                                    logging.info(f"扫描器 (并发): 新LLM任务已入队: {task_id}")
                            else:
                                if not app_state["llm_tasks"]:
                                    if has_vad and not has_asr and task_id not in app_state["asr_tasks"]:
                                        app_state["asr_tasks"][task_id] = "排队中"
                                        await app_state["asr_queue"].put(task_id)
                                        logging.info(f"扫描器 (顺序): 新ASR任务已入队: {task_id}")
                                
                                if not app_state["asr_tasks"]:
                                    if has_vad and has_asr and task_id not in app_state["llm_tasks"]:
                                        app_state["llm_tasks"][task_id] = "排队中"
                                        await app_state["llm_queue"].put(task_id)
                                        logging.info(f"扫描器 (顺序): 新LLM任务已入队: {task_id}")
            except Exception as e:
                logging.error(f"扫描器: 扫描循环出错: {e}")
        await asyncio.sleep(5)

async def asr_worker():
    cfg = app_state["config"]
    api_url = cfg["remote_asr_api_url"]
    api_params = cfg["asr_api_params"]
    async with httpx.AsyncClient(timeout=None) as client:
        while True:
            task_id = await app_state["asr_queue"].get()
            logging.info(f"ASR Worker: 开始处理任务 {task_id}")
            app_state["asr_tasks"][task_id] = "进行中"
            try:
                vad_info_path = os.path.join(task_id, "_vad_complete.json")
                with open(vad_info_path, 'r', encoding='utf-8') as f: vad_info = json.load(f)
                files_to_upload = [("files", (os.path.basename(c["file_path"]), open(c["file_path"], "rb"), "audio/wav")) for c in vad_info]
                response = await client.post(api_url, data=api_params, files=files_to_upload)
                for _, file_tuple in files_to_upload: file_tuple[1].close()
                if response.status_code == 200:
                    results = response.json()
                    results_dict = {r['uttid']: r['text'] for r in results}
                    sorted_transcripts = [{"file": c["file_path"], "text": results_dict.get(os.path.splitext(os.path.basename(c["file_path"]))[0], "[识别失败]")} for c in vad_info]
                    with open(os.path.join(task_id, "_asr_complete.json"), 'w', encoding='utf-8') as f: json.dump(sorted_transcripts, f, indent=4, ensure_ascii=False)
                    app_state["asr_tasks"][task_id] = "完成"
                    logging.info(f"ASR Worker: 任务 {task_id} 成功完成。")
                    if cfg.get("allow_concurrent_asr_llm", False) and task_id not in app_state["llm_tasks"]:
                       app_state["llm_tasks"][task_id] = "排队中"
                       await app_state["llm_queue"].put(task_id)
                    await asyncio.sleep(1) 
                    app_state["asr_tasks"].pop(task_id, None)
                else:
                    logging.error(f"ASR Worker: API请求失败 ({response.status_code}): {response.text}")
                    app_state["asr_tasks"][task_id] = f"失败: API错误 {response.status_code}"
            except Exception as e:
                logging.error(f"ASR Worker: 处理任务 {task_id} 时出错: {e}")
                app_state["asr_tasks"][task_id] = f"失败: {e}"
            app_state["asr_queue"].task_done()

async def llm_worker():
    cfg = app_state["config"]
    client = app_state["openai_client"]
    semaphore = asyncio.Semaphore(cfg["llm_concurrency"])
    async def process_chunk(chunk_text: str) -> str:
        async with semaphore:
            try:
                prompt = f"""你是一个专业的速记员和文本后期处理专家。请将以下由语音识别（ASR）生成的原始文本进行处理。你的任务是：
1.  修正明显的识别错误。
2.  添加恰当的标点符号，包括逗号、句号、问号等。
3.  根据上下文和语义，将文本进行合理的分段，用换行符分隔。
请直接返回处理后的文本，不要包含任何额外的解释或标题。

原始文本如下：
"{chunk_text}"
"""
                completion = await client.chat.completions.create(model=cfg["llm_model_name"], messages=[{"role": "user", "content": prompt}], temperature=0.3)
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"LLM Worker: 调用API时出错: {e}")
                return f"[LLM处理失败: {e}]"
    while True:
        task_id = await app_state["llm_queue"].get()
        logging.info(f"LLM Worker: 开始处理任务 {task_id}")
        app_state["llm_tasks"][task_id] = "进行中"
        try:
            with open(os.path.join(task_id, "_asr_complete.json"), 'r', encoding='utf-8') as f: transcripts = [item['text'] for item in json.load(f)]
            chunks, current_chunk, current_len = [], [], 0
            for i, text in enumerate(transcripts):
                current_chunk.append(text)
                current_len += len(text)
                if current_len >= 500 and i < len(transcripts) - 1:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [current_chunk[-1]]
                    current_len = len(current_chunk[0])
            if current_chunk: chunks.append(" ".join(current_chunk))
            llm_tasks = [process_chunk(chunk) for chunk in chunks]
            llm_results = await asyncio.gather(*llm_tasks)
            final_text = merge_llm_results(llm_results)
            output_txt_path = os.path.join(os.path.dirname(task_id), f"{os.path.basename(task_id)}.txt")
            with open(output_txt_path, 'w', encoding='utf-8') as f: f.write(final_text)
            await run_in_threadpool(shutil.rmtree, task_id)
            app_state["llm_tasks"][task_id] = "完成"
            logging.info(f"LLM Worker: 任务 {task_id} 完成。")
            await asyncio.sleep(1)
            app_state["llm_tasks"].pop(task_id, None)
        except Exception as e:
            logging.error(f"LLM Worker: 处理任务 {task_id} 时出错: {e}")
            app_state["llm_tasks"][task_id] = f"失败: {e}"
        app_state["llm_queue"].task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["config"] = load_config()
    app_state["audio_splitter"] = AudioSplitter()
    app_state["openai_client"] = AsyncOpenAI(api_key=app_state["config"]["openai_api_key"], base_url=app_state["config"]["openai_api_base"])
    
    task0_scanner = asyncio.create_task(vad_scan_loop())
    task0_worker = asyncio.create_task(vad_worker())
    task1_scanner = asyncio.create_task(transcription_scan_loop())
    task2_worker = asyncio.create_task(asr_worker())
    task3_worker = asyncio.create_task(llm_worker())
    app_state["background_tasks"].extend([task0_scanner, task0_worker, task1_scanner, task2_worker, task3_worker])
    
    logging.info("所有后台任务已启动。")
    yield
    logging.info("正在关闭应用...")
    for task in app_state["background_tasks"]: task.cancel()
    await asyncio.gather(*app_state["background_tasks"], return_exceptions=True)
    logging.info("所有后台任务已取消。")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root(): return FileResponse('static/index.html')

@app.get("/api/queues")
async def get_queues_status():
    return {
        "scanning_active": app_state["scanning_active"],
        "vad_queue": { "tasks": app_state["vad_tasks"], "count": app_state["vad_queue"].qsize()},
        "asr_queue": { "tasks": [{"id": k, "status": v} for k, v in app_state["asr_tasks"].items()], "count": app_state["asr_queue"].qsize()},
        "llm_queue": { "tasks": [{"id": k, "status": v} for k, v in app_state["llm_tasks"].items()], "count": app_state["llm_queue"].qsize()}
    }

@app.post("/api/start_transcription")
async def start_transcription():
    if not app_state["scanning_active"]:
        app_state["scanning_active"] = True
        return {"message": "转录扫描已启动"}
    return {"message": "转录扫描已在运行中"}

@app.post("/api/stop_transcription")
async def stop_transcription():
    if app_state["scanning_active"]:
        app_state["scanning_active"] = False
        while not app_state["asr_queue"].empty():
            task_id = await app_state["asr_queue"].get()
            app_state["asr_tasks"].pop(task_id, None)
            app_state["asr_queue"].task_done()
        while not app_state["llm_queue"].empty():
            task_id = await app_state["llm_queue"].get()
            app_state["llm_tasks"].pop(task_id, None)
            app_state["llm_queue"].task_done()
        for task_id, status in app_state["asr_tasks"].items():
            if status == "排队中": app_state["asr_tasks"][task_id] = "已取消"
        for task_id, status in app_state["llm_tasks"].items():
            if status == "排队中": app_state["llm_tasks"][task_id] = "已取消"
        return {"message": "转录扫描已停止"}
    return {"message": "转录扫描已经停止"}

app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=33013)