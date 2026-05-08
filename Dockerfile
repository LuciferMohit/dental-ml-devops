FROM python:3.9-slim
   
   WORKDIR /app
   
   RUN apt-get update && apt-get install -y \
       libglib2.0-0 libsm6 libxext6 libxrender-dev \
       && rm -rf /var/lib/apt/lists/*
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   # Note: I updated the filename here to match what you named it
   CMD ["streamlit", "run", "devops_ver_onnx.py", "--server.port=8501", "--server.address=0.0.0.0"]