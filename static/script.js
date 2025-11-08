// ClassTranscribe/static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const transcribeBtn = document.getElementById('transcribe-btn');
    const scanStatus = document.getElementById('scan-status');

    const vadList = document.getElementById('vad-list');
    const asrList = document.getElementById('asr-list');
    const llmList = document.getElementById('llm-list');
    
    const vadCount = document.getElementById('vad-count');
    const asrCount = document.getElementById('asr-count');
    const llmCount = document.getElementById('llm-count');

    let isScanning = false;
    let pollInterval;

    function setButtonState(scanning) {
        isScanning = scanning;
        if (isScanning) {
            transcribeBtn.textContent = '停止转录';
            transcribeBtn.className = 'stop';
            scanStatus.textContent = '扫描状态: 运行中';
        } else {
            transcribeBtn.textContent = '扫描并转录';
            transcribeBtn.className = 'start';
            scanStatus.textContent = '扫描状态: 停止';
        }
    }

    async function toggleTranscription() {
        const endpoint = isScanning ? '/api/stop_transcription' : '/api/start_transcription';
        try {
            const response = await fetch(endpoint, { method: 'POST' });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log(data.message);
            // 立即更新状态，而不是等待下一次轮询
            setButtonState(!isScanning);
        } catch (error) {
            console.error("Error toggling transcription:", error);
        }
    }

    function renderTaskList(element, tasks, isSimple = false) {
        element.innerHTML = '';
        if (tasks.length === 0) {
            element.innerHTML = '<li class="task-item">队列为空</li>';
            return;
        }
        tasks.forEach(task => {
            const li = document.createElement('li');
            li.className = 'task-item';
            
            const taskId = isSimple ? task.file : task.id.split('/').pop();
            const statusClass = `status-${task.status}`;

            li.innerHTML = `
                <span class="id">${taskId}</span>
                <span class="status-label ${statusClass}">${task.status}</span>
            `;
            element.appendChild(li);
        });
    }

    async function pollQueues() {
        try {
            const response = await fetch('/api/queues');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            setButtonState(data.scanning_active);

            vadCount.textContent = `(${data.vad_queue.count})`;
            asrCount.textContent = `(${data.asr_queue.count} / ${data.asr_queue.tasks.length})`;
            llmCount.textContent = `(${data.llm_queue.count} / ${data.llm_queue.tasks.length})`;

            renderTaskList(vadList, data.vad_queue.tasks, true);
            renderTaskList(asrList, data.asr_queue.tasks.reverse());
            renderTaskList(llmList, data.llm_queue.tasks.reverse());

        } catch (error) {
            console.error("Error polling queues:", error);
            // 如果出错，停止轮询
            clearInterval(pollInterval);
        }
    }

    transcribeBtn.addEventListener('click', toggleTranscription);

    // 初始加载并开始轮询
    pollQueues();
    pollInterval = setInterval(pollQueues, 500); // 每0.5秒轮询一次
});