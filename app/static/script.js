document.addEventListener('DOMContentLoaded', function() {
    const vadProgress = document.getElementById('vad-progress');
    const asrProgress = document.getElementById('asr-progress');
    const mergeProgress = document.getElementById('merge-progress');
    
    const vadTaskInfo = document.getElementById('vad-task-info');
    const asrTaskInfo = document.getElementById('asr-task-info');
    const mergeTaskInfo = document.getElementById('merge-task-info');

    const actionButton = document.getElementById('action-button');
    const buttonText = document.getElementById('button-text');
    const buttonSpinner = document.getElementById('button-spinner');
    const actionStatus = document.getElementById('action-status');

    function updateStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                const vadQueueLen = data.vad_queue_len || 0;
                const asrQueueLen = data.asr_queue_len || 0;
                const mergeQueueLen = data.merge_queue_len || 0;

                // 更新进度条文本
                vadProgress.textContent = `${vadQueueLen}`;
                asrProgress.textContent = `${asrQueueLen}`;
                mergeProgress.textContent = `${mergeQueueLen}`;

                // 进度条宽度简单表示队列是否为空
                vadProgress.style.width = vadQueueLen > 0 ? '100%' : '0%';
                asrProgress.style.width = asrQueueLen > 0 ? '100%' : '0%';
                mergeProgress.style.width = mergeQueueLen > 0 ? '100%' : '0%';

                // 更新当前任务信息
                vadTaskInfo.textContent = `当前任务: ${data.current_vad_task}`;
                asrTaskInfo.textContent = `当前任务: ${data.current_asr_task}`;
                mergeTaskInfo.textContent = `当前任务: ${data.current_merge_task}`;
            })
            .catch(error => {
                console.error('获取状态失败:', error);
                vadTaskInfo.textContent = "与后端连接断开...";
                asrTaskInfo.textContent = "";
                mergeTaskInfo.textContent = "";
            });
    }
    
    actionButton.addEventListener('click', () => {
        buttonText.textContent = "正在请求...";
        buttonSpinner.classList.remove('d-none');
        actionButton.disabled = true;
        actionStatus.textContent = "";

        fetch('/start-asr-llm-tasks', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                actionStatus.textContent = data.message || "未知响应";
                setTimeout(() => { actionStatus.textContent = ""; }, 5000);
            })
            .catch(error => {
                console.error('启动任务失败:', error);
                actionStatus.textContent = "请求失败，请检查后端服务是否正常。";
            })
            .finally(() => {
                buttonText.textContent = "扫描并启动 ASR 与合并任务";
                buttonSpinner.classList.add('d-none');
                actionButton.disabled = false;
            });
    });

    // 初始加载并每2秒更新一次
    updateStatus();
    setInterval(updateStatus, 2000);
});