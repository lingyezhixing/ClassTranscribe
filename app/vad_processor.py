import os
import threading
import time
from .state_manager import state_manager

# --- 将 VAD_example.py 中的 AudioSplitter 类代码完整粘贴到这里 ---
# ... (为节省篇幅，此处省略，请将您提供的 VAD_example.py 中的 AudioSplitter 类完整复制于此)
# 为了确保代码独立可运行，这里提供一个精简版的占位符，您需要用真实代码替换它
class AudioSplitter:
    def __init__(self, use_onnx: bool = True):
        print("VAD 处理器已初始化 (这是一个占位符，请使用您提供的完整代码)")
    def split(self, audio_path, output_dir, **kwargs):
        print(f"正在对 {audio_path} 进行VAD切分 (占位符)...")
        os.makedirs(output_dir, exist_ok=True)
        # 模拟操作
        time.sleep(5)
        with open(os.path.join(output_dir, "placeholder_chunk_0001.wav"), "w") as f:
            f.write("fake audio")
        print(f"VAD切分完成: {audio_path}")

# --- ---

def process_vad_queue(norm_processes: int):
    """一个在后台线程中运行的函数，持续处理VAD队列中的任务。"""
    splitter = AudioSplitter()
    while True:
        try:
            if state_manager.vad_queue:
                filepath = state_manager.vad_queue.popleft()
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                output_dir = os.path.join(os.path.dirname(filepath), base_name)
                
                state_manager.current_vad_task = f"正在切分: {base_name}"
                print(f"开始VAD任务: {filepath}")
                
                splitter.split(
                    audio_path=filepath,
                    output_dir=output_dir,
                    target_length=13,
                    max_length=15,
                    overlap_length=1,
                    skip_vad=False,
                    normalize_audio=True,
                    norm_processes=norm_processes
                )
                
                print(f"VAD任务完成: {filepath}")
                state_manager.current_vad_task = "空闲"
            else:
                time.sleep(5) # 队列为空时，等待5秒
        except Exception as e:
            print(f"VAD处理线程出错: {e}")
            state_manager.current_vad_task = f"错误: {e}"
            time.sleep(10) # 出错后等待更长时间