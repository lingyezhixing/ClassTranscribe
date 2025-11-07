import yaml
import os
import threading
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.background import BackgroundScheduler

from .state_manager import state_manager
from .scheduler import scan_and_move_files
from .vad_processor import process_vad_queue
from .asr_llm_processor import Processor

# --- 全局变量 ---
config = {}
scheduler = BackgroundScheduler()
processor: Processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 应用启动时执行 ---
    global config, processor
    print("应用启动中...")
    
    # 1. 加载配置
    with open("app/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("配置加载成功。")
    
    # 2. 初始化后台处理器
    processor = Processor(config)
    
    # 3. 启动后台处理线程/任务
    # VAD队列处理 (在独立线程中运行，因为它可能包含阻塞的CPU密集型操作)
    vad_thread = threading.Thread(
        target=process_vad_queue, 
        args=(config['vad_norm_processes'],), 
        daemon=True
    )
    vad_thread.start()
    
    # ASR 和 LLM 队列处理 (在asyncio事件循环中运行)
    asyncio.create_task(processor.process_asr_queue())
    asyncio.create_task(processor.process_llm_merge_queue())
    
    print("后台处理任务已启动。")
    
    # 4. 启动定时扫描任务
    scheduler.add_job(
        scan_and_move_files,
        'interval',
        minutes=config['scan_interval_minutes'],
        args=[config['path_pairs']]
    )
    scheduler.start()
    print(f"定时扫描任务已启动，每 {config['scan_interval_minutes']} 分钟执行一次。")
    
    yield
    
    # --- 应用关闭时执行 ---
    scheduler.shutdown()
    print("定时扫描任务已关闭。")

# --- FastAPI 应用实例 ---
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="app/templates")

# --- API 端点 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """渲染主页面"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    """获取后端当前状态，用于前端轮询"""
    return state_manager.get_state()

@app.post("/start-asr-llm-tasks")
async def start_asr_llm_tasks():
    """扫描已完成VAD的目录，并将它们加入ASR处理队列"""
    print("收到前端请求：开始ASR与合并任务...")
    found_tasks = 0
    for pair in config.get('path_pairs', []):
        transfer_path = pair['transfer_path']
        if not os.path.isdir(transfer_path):
            continue
        
        for item in os.listdir(transfer_path):
            item_path = os.path.join(transfer_path, item)
            if os.path.isdir(item_path): # VAD切分后的文件保存在同名目录中
                # 检查是否已处理：是否存在 final_transcript.txt
                if not os.path.exists(os.path.join(item_path, "final_transcript.txt")):
                    if item_path not in state_manager.asr_queue:
                         state_manager.asr_queue.append(item_path)
                         found_tasks += 1
    
    if found_tasks > 0:
        return {"message": f"成功添加 {found_tasks} 个新任务到ASR队列。"}
    else:
        return {"message": "没有找到新的可处理任务。"}