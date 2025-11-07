import threading
from collections import deque
from typing import Deque, List, Dict, Any

class StateManager:
    """线程安全的单例模式，用于管理整个应用的状态。"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 使用 set 来确保只初始化一次
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not hasattr(self, '_initialized'):
                    self.vad_queue: Deque[str] = deque()
                    self.asr_queue: Deque[str] = deque()
                    self.merge_queue: Deque[Dict[str, Any]] = deque()
                    
                    self.vad_total_count = 0
                    self.asr_total_count = 0
                    self.merge_total_count = 0
                    
                    self.current_vad_task = "空闲"
                    self.current_asr_task = "空闲"
                    self.current_merge_task = "空闲"
                    
                    self._initialized = True

    def add_to_vad_queue(self, path: str):
        with self._lock:
            if path not in self.vad_queue:
                self.vad_queue.append(path)

    def get_state(self) -> Dict[str, Any]:
        """获取当前所有状态的快照。"""
        with self._lock:
            return {
                "vad_queue_len": len(self.vad_queue),
                "asr_queue_len": len(self.asr_queue),
                "merge_queue_len": len(self.merge_queue),
                "current_vad_task": self.current_vad_task,
                "current_asr_task": self.current_asr_task,
                "current_merge_task": self.current_merge_task,
            }

state_manager = StateManager()