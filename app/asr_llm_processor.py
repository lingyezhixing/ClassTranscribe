import os
import json
import asyncio
import aiohttp
import time
import re
from openai import AsyncOpenAI
from .state_manager import state_manager

class Processor:
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config['openai_api_key'],
            base_url=config['openai_api_base']
        )
        self.llm_semaphore = asyncio.Semaphore(config['llm_concurrency'])

    async def process_asr_queue(self):
        """后台任务：处理ASR任务队列"""
        while True:
            if state_manager.asr_queue:
                folder_path = state_manager.asr_queue.popleft()
                folder_name = os.path.basename(folder_path)
                state_manager.current_asr_task = f"ASR: {folder_name}"
                print(f"开始ASR任务: {folder_path}")

                try:
                    # 1. 查找所有音频切片
                    chunk_files = sorted([
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path) if f.endswith('.wav')
                    ])
                    if not chunk_files:
                        print(f"警告: 目录 {folder_path} 中没有找到.wav切片，跳过。")
                        continue

                    # 2. 调用远程ASR API
                    async with aiohttp.ClientSession() as session:
                        form_data = aiohttp.FormData()
                        for param, value in self.config['asr_api_params'].items():
                            if value is not None:
                                form_data.add_field(param, str(value))
                        
                        for f_path in chunk_files:
                            form_data.add_field(
                                'files',
                                open(f_path, 'rb'),
                                filename=os.path.basename(f_path),
                                content_type='audio/wav'
                            )
                        
                        api_url = self.config['remote_asr_api_url']
                        async with session.post(api_url, data=form_data) as response:
                            if response.status == 200:
                                asr_results = await response.json()
                                # 保存原始ASR结果
                                with open(os.path.join(folder_path, "asr_results.json"), "w", encoding='utf-8') as f:
                                    json.dump(asr_results, f, ensure_ascii=False, indent=2)
                                print(f"ASR成功: {folder_name}")
                                # 3. 加入合并队列
                                state_manager.merge_queue.append({
                                    "folder_path": folder_path,
                                    "results": asr_results
                                })
                            else:
                                error_text = await response.text()
                                print(f"ASR API错误 for {folder_name}: {response.status} - {error_text}")

                except Exception as e:
                    print(f"ASR处理失败: {folder_name}, 错误: {e}")
                
                state_manager.current_asr_task = "空闲"
            else:
                await asyncio.sleep(5)

    async def process_llm_merge_queue(self):
        """后台任务：处理LLM合并任务队列"""
        while True:
            if state_manager.merge_queue:
                task = state_manager.merge_queue.popleft()
                folder_path = task['folder_path']
                results = task['results']
                folder_name = os.path.basename(folder_path)
                state_manager.current_merge_task = f"合并: {folder_name}"
                print(f"开始LLM合并任务: {folder_name}")

                try:
                    # 1. 准备文本片段
                    sorted_results = sorted(results, key=lambda x: x['uttid'])
                    transcripts = [res['text'] for res in sorted_results if res.get('text')]
                    
                    if not transcripts:
                        print(f"警告: {folder_name} 没有有效的转录文本，跳过合并。")
                        continue
                    
                    # 2. 按约500字分块，并有1条重叠
                    text_chunks = []
                    current_chunk = []
                    current_len = 0
                    for i, text in enumerate(transcripts):
                        current_chunk.append(text)
                        current_len += len(text)
                        if current_len >= 500 and i < len(transcripts) - 1:
                            text_chunks.append("\n".join(current_chunk))
                            current_chunk = [text] # 下一个块包含上一块的最后一条
                            current_len = len(text)
                    if current_chunk:
                        text_chunks.append("\n".join(current_chunk))

                    # 3. 并发调用LLM API
                    llm_tasks = [self.refine_text_chunk(chunk, i) for i, chunk in enumerate(text_chunks)]
                    refined_chunks = await asyncio.gather(*llm_tasks)
                    refined_chunks.sort(key=lambda x: x[0]) # 按索引排序

                    # 4. 合并重叠部分
                    final_text = self.merge_overlapping_chunks([chunk for _, chunk in refined_chunks])

                    # 5. 保存最终结果
                    with open(os.path.join(folder_path, "final_transcript.txt"), "w", encoding='utf-8') as f:
                        f.write(final_text)
                    print(f"LLM合并成功: {folder_name}")

                except Exception as e:
                    print(f"LLM合并失败: {folder_name}, 错误: {e}")
                
                state_manager.current_merge_task = "空闲"

            else:
                await asyncio.sleep(5)
    
    async def refine_text_chunk(self, text: str, index: int):
        async with self.llm_semaphore:
            print(f"向LLM发送第 {index+1} 个文本块...")
            prompt = (
                "你是一个专业的速记员和文本后期处理专家。"
                "以下是一段由语音识别系统生成的原始文字，可能包含错误、缺少标点和不合理的分段。"
                "请你完成以下任务：\n"
                "1. 修正明显的识别错误。\n"
                "2. 添加恰当的标点符号，包括逗号、句号、问号等。\n"
                "3. 根据语义和上下文，将文本进行合理的分段，使用换行符分隔段落。\n"
                "4. 除了修正和格式化，不要添加任何额外的内容、评论或总结。\n"
                "5. 直接返回处理后的文本。\n\n"
                f"原始文本：\n---\n{text}\n---\n处理后的文本："
            )
            try:
                response = await self.client.chat.completions.create(
                    model=self.config['llm_model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return (index, response.choices[0].message.content)
            except Exception as e:
                print(f"LLM API调用失败 (块 {index+1}): {e}")
                return (index, f"【LLM处理失败：{e}】\n\n" + text) # 返回原始内容以防数据丢失

    def merge_overlapping_chunks(self, chunks: list[str]) -> str:
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0]

        final_text = chunks[0]
        for i in range(len(chunks) - 1):
            prev_chunk = final_text
            next_chunk = chunks[i+1]
            
            # 寻找一个好的合并点，从重叠部分的末尾开始向前找一个标点
            overlap_len = len(prev_chunk) // 2 # 粗略估计重叠区域
            search_area = prev_chunk[-overlap_len:]
            
            split_indices = [m.start() for m in re.finditer(r'[。？！]\s*', search_area)]
            if split_indices:
                split_point = len(prev_chunk) - overlap_len + split_indices[-1] + 1
                final_text = prev_chunk[:split_point]
                
                # 在下一块中找到匹配的开头并裁剪
                match_start = next_chunk.find(prev_chunk[split_point:].strip())
                if match_start != -1:
                    next_chunk_trimmed = next_chunk[match_start + len(prev_chunk[split_point:].strip()):].strip()
                    final_text += "\n" + next_chunk_trimmed
                else:
                    final_text += "\n" + next_chunk # 如果找不到，直接拼接
            else:
                 final_text += "\n" + next_chunk # 如果没有标点，直接拼接
        
        return final_text