FROM python:3.10-slim

WORKDIR /app

#Cài các gói cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt /app/requirements.txt


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

#Cổng mạng mà ứng dụng sẽ mở ra để Streamlit có thể truy cập từ bên ngoài container
EXPOSE 8501

CMD ["streamlit", "run", "src/app.py","--server.port=8501", "--server.address=0.0.0.0"]