# ClassTranscribe/Dockerfile

# 1. 使用官方的 Python 3.12-slim 作为基础镜像
FROM python:3.12-slim

# 2. 在容器内设置工作目录
WORKDIR /app

# 3. 更新包列表并安装 ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 安装 Python 依赖，--no-cache-dir 减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 6. 将项目中的所有文件复制到工作目录
COPY . .

# 7. 声明应用运行的端口
EXPOSE 33013

# 8. 容器启动时运行的命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "33013"]