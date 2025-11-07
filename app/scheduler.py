import os
import shutil
import re
from datetime import datetime
from .state_manager import state_manager

def scan_and_move_files(path_pairs: list):
    print("定时任务：开始扫描新文件...")
    date_pattern = re.compile(r"^\d{4}年\d{2}月\d{2}日")
    today_str = datetime.now().strftime("%Y年%m月%d日")

    for pair in path_pairs:
        scan_path = pair['scan_path']
        transfer_path = pair['transfer_path']
        os.makedirs(scan_path, exist_ok=True)
        os.makedirs(transfer_path, exist_ok=True)

        for filename in os.listdir(scan_path):
            original_filepath = os.path.join(scan_path, filename)
            if not os.path.isfile(original_filepath):
                continue
            
            base, ext = os.path.splitext(filename)
            
            # 1. 检查并重命名
            if not date_pattern.match(base):
                new_base = today_str
            else:
                new_base = base
            
            # 2. 检查转移路径下的冲突并处理
            counter = 0
            final_base_name = new_base
            
            while True:
                conflict_found = False
                for existing_file in os.listdir(transfer_path):
                    if existing_file.startswith(final_base_name):
                        conflict_found = True
                        break
                
                if not conflict_found:
                    break
                    
                counter += 1
                final_base_name = f"{new_base}-{counter}"

            # 3. 确定最终文件名并移动
            if counter > 0:
                final_scan_base = f"{new_base}-{counter+1}"
                final_transfer_base = final_base_name
            else:
                final_scan_base = new_base
                final_transfer_base = new_base

            final_scan_name = final_scan_base + ext
            final_transfer_name = final_transfer_base + ext
            
            # 重命名扫描路径下的文件
            final_scan_path = os.path.join(scan_path, final_scan_name)
            os.rename(original_filepath, final_scan_path)
            
            # 移动到转移路径
            destination_path = os.path.join(transfer_path, final_transfer_name)
            shutil.move(final_scan_path, destination_path)
            
            print(f"文件处理完成: '{filename}' -> '{final_transfer_name}' 并移动至 '{transfer_path}'")
            
            # 4. 加入VAD队列
            state_manager.add_to_vad_queue(destination_path)