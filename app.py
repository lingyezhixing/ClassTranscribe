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
import base64
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from AudioSplitter import AudioSplitter

# --- 新的两阶段提示词 ---
PROMPT_STAGE_1 = """你是一位顶级的速记员和文本编辑。你将收到一段由语音识别（ASR）直接生成的原始文本，其中包含了许多因强制切分导致的不完整的短句，每行一个。

你的任务是：
1. **智能合并**：将这些零碎的短语无缝地拼接成通顺、完整的句子。
2. **添加标点**：为全文添加精确的标点符号，包括逗号、句号、问号等，使文本更具可读性。
3. **修正错误**：根据上下文，修正ASR可能产生的明显识别错误（如同音字、错别字）。
4. **合理分段**：在适当的地方进行换行，形成逻辑清晰的段落。

**要求**：
- 直接返回处理后的文本。
- **不要**包含任何解释、标题或“处理后文本：”这样的前缀。

原始文本碎片如下：
---
{chunk_text}
---

处理后的文本："""

PROMPT_STAGE_2 = """你是一位语言大师，擅长优化文本的衔接与流畅度。我遇到了文本的中间存在一个因技术原因造成的、不自然的断裂或转折点的问题，我已经从断裂点处分割了前后段落，请你进行处理。

你的**唯一任务**是：
- **重写整个文本段**，使其成为一个单一、连贯、流畅的整体。
- 确保上下文逻辑通顺，完美地弥合中间的断裂感。
- 保留原始的核心信息，但要让过渡变得无法察觉。

**要求**：
- 直接返回重写后的、无缝衔接的完整文本段。
- **不要**添加任何与原文无关的评论或解释。

断点前的文本：{prev_junction_text}

断点后的文本：{next_junction_text}

重写后的完整文本："""


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

                    # 同步重命名对应的文件夹
                    old_dir_path = os.path.join(transfer_path, existing_base_name)
                    new_indexed_dir_name = f"{base_name_A}-1"
                    new_dir_path = os.path.join(transfer_path, new_indexed_dir_name)

                    if await run_in_threadpool(os.path.isdir, old_dir_path):
                        await run_in_threadpool(os.rename, old_dir_path, new_dir_path)
                        logging.info(f"VAD Worker: 同步重命名文件夹: {existing_base_name} -> {new_indexed_dir_name}")
                
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
    # 基础参数，不包含动态生成的部分
    base_api_params = cfg["asr_api_params"]

    # --- 辅助函数：在线程中进行文件读取和Base64编码，防止阻塞事件循环 ---
    def get_base64_encoded_audio(file_path: str) -> str:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')

    async with httpx.AsyncClient(timeout=None) as client:
        while True:
            task_id = await app_state["asr_queue"].get()
            logging.info(f"ASR Worker: 开始处理任务 {task_id}")
            app_state["asr_tasks"][task_id] = "进行中"
            
            try:
                vad_info_path = os.path.join(task_id, "_vad_complete.json")
                with open(vad_info_path, 'r', encoding='utf-8') as f:
                    vad_info = json.load(f)

                # --- 核心改动开始: 构建JSON请求体 ---

                # 1. 异步地将所有音频文件编码为Base64
                audio_sources = []
                for chunk in vad_info:
                    file_path = chunk["file_path"]
                    encoded_data = await run_in_threadpool(get_base64_encoded_audio, file_path)
                    audio_sources.append({
                        "file_name": os.path.basename(file_path),
                        "audio_data": encoded_data
                    })
                
                # 2. 构造最终的JSON Payload
                payload = base_api_params.copy()
                payload["audio_files"] = audio_sources
                payload["stream"] = False # 添加API要求的固定参数

                # 3. 发送JSON请求
                logging.info(f"ASR Worker: 正在向 {api_url} 发送包含 {len(audio_sources)} 个文件的JSON请求...")
                response = await client.post(api_url, json=payload)
                
                # --- 核心改动结束 ---

                if response.status_code == 200:
                    results = response.json()
                    # 响应处理逻辑保持不变
                    results_dict = {r['uttid']: r['text'] for r in results}
                    sorted_transcripts = [
                        {
                            "file": c["file_path"],
                            "text": results_dict.get(os.path.splitext(os.path.basename(c["file_path"]))[0], "[识别失败]")
                        } for c in vad_info
                    ]
                    
                    with open(os.path.join(task_id, "_asr_complete.json"), 'w', encoding='utf-8') as f:
                        json.dump(sorted_transcripts, f, indent=4, ensure_ascii=False)
                        
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
                logging.error(f"ASR Worker: 处理任务 {task_id} 时出错: {e}", exc_info=True)
                app_state["asr_tasks"][task_id] = f"失败: {e}"
            
            app_state["asr_queue"].task_done()

async def llm_worker():
    cfg = app_state["config"]
    client = app_state["openai_client"]
    semaphore = asyncio.Semaphore(cfg["llm_concurrency"])
    PUNCTUATION_SEARCH = re.compile(r'[.!?。！？\n]')

    async def process_llm_request(prompt: str, model: str, temperature: float) -> str:
        async with semaphore:
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"LLM Worker: 调用API时出错: {e}")
                return f"[LLM处理失败: {e}]"

    while True:
        task_id = await app_state["llm_queue"].get()
        logging.info(f"LLM Worker: 开始处理任务 {task_id}")
        app_state["llm_tasks"][task_id] = "进行中"
        try:
            asr_complete_path = os.path.join(task_id, "_asr_complete.json")
            with open(asr_complete_path, 'r', encoding='utf-8') as f:
                transcripts = [item['text'] for item in json.load(f)]

            # --- 阶段一: 将ASR碎片整合为连贯的文本块 ---
            logging.info(f"LLM Worker (任务 {task_id}): 开始阶段一处理...")
            chunks, current_chunk_lines, current_len = [], [], 0
            for text in transcripts:
                current_chunk_lines.append(text)
                current_len += len(text)
                if current_len >= 500:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines, current_len = [], 0
            if current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))

            stage1_prompts = [PROMPT_STAGE_1.format(chunk_text=chunk) for chunk in chunks]
            stage1_tasks = [process_llm_request(p, cfg["llm_model_name"], 0.3) for p in stage1_prompts]
            stage1_results = await asyncio.gather(*stage1_tasks)
            logging.info(f"LLM Worker (任务 {task_id}): 阶段一完成，生成 {len(stage1_results)} 个文本块。")

            # --- 阶段二: 平滑处理文本块之间的连接处 ---
            if not stage1_results:
                final_text = ""
            elif len(stage1_results) == 1:
                final_text = stage1_results[0]
            else:
                logging.info(f"LLM Worker (任务 {task_id}): 开始阶段二处理，平滑 {len(stage1_results) - 1} 个连接点...")
                text_blocks = list(stage1_results)
                
                for i in range(len(text_blocks) - 1):
                    prev_block = text_blocks[i]
                    next_block = text_blocks[i+1]

                    # 从前一个块的末尾向前取约250个字符作为上下文
                    slice_len_prev = min(len(prev_block), 250)
                    search_area_prev = prev_block[-slice_len_prev:]
                    matches_prev = list(PUNCTUATION_SEARCH.finditer(search_area_prev))
                    # 在上下文中寻找第一个标点作为切分点，以保证句子完整性
                    if matches_prev:
                        # 修正点: 之前是 matches_prev[-1]，现在改为 matches_prev[0]，以实现“向前找第一个标点”
                        split_offset = matches_prev[0].end()
                        split_point_prev = len(prev_block) - slice_len_prev + split_offset
                    else:
                        split_point_prev = len(prev_block) - slice_len_prev

                    # 从后一个块的开头向后取约250个字符作为上下文
                    slice_len_next = min(len(next_block), 250)
                    search_area_next = next_block[:slice_len_next]
                    matches_next = list(PUNCTUATION_SEARCH.finditer(search_area_next))
                    # 在上下文中寻找最后一个标点作为切分点 (此逻辑是正确的)
                    if matches_next:
                        split_point_next = matches_next[-1].end()
                    else:
                        split_point_next = slice_len_next
                    
                    prev_junction_text = prev_block[split_point_prev:]
                    next_junction_text = next_block[:split_point_next]

                    # 调用LLM平滑连接
                    stage2_prompt = PROMPT_STAGE_2.format(
                        prev_junction_text=prev_junction_text,
                        next_junction_text=next_junction_text
                    )
                    smoothed_junction = await process_llm_request(stage2_prompt, cfg["llm_model_name"], 0.3)
                    
                    # 用返回结果替换原来的连接部分
                    text_blocks[i] = prev_block[:split_point_prev]
                    # 关键: 将下一个块的全部内容替换为 平滑段 + 下一个块的剩余部分
                    text_blocks[i+1] = smoothed_junction + next_block[split_point_next:]
                    logging.info(f"LLM Worker (任务 {task_id}): 已处理第 {i+1}/{len(stage1_results) - 1} 个连接点。")

                final_text = "".join(text_blocks)

            # --- 保存最终结果并清理 ---
            output_txt_path = os.path.join(os.path.dirname(task_id), f"{os.path.basename(task_id)}.txt")
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            
            await run_in_threadpool(shutil.rmtree, task_id)
            app_state["llm_tasks"][task_id] = "完成"
            logging.info(f"LLM Worker: 任务 {task_id} 完成，最终文稿已保存至 {output_txt_path}")
            await asyncio.sleep(1)
            app_state["llm_tasks"].pop(task_id, None)

        except Exception as e:
            logging.error(f"LLM Worker: 处理任务 {task_id} 时出错: {e}", exc_info=True)
            app_state["llm_tasks"][task_id] = f"失败: {e}"
        finally:
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