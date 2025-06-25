# Use a minimal Python image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY ../../requirements.txt .

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    libc6-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libopencv-dev \
    libatlas-base-dev \
    liblapack-dev \
    libopenblas-dev \     
    git \  
    ccache \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"


RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code into the container
COPY ../../src ./src

# Set environment variables
ENV PYTHONPATH=/app/src

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "src/UI/main.py", "--server.headless=true", "--server.port=8501", "--server.enableCORS=false"]
