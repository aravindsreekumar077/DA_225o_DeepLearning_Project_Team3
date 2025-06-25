FROM python:3.11-slim
LABEL authors="Manoj Kokane"
WORKDIR /app


COPY requirements.txt .


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

RUN pip install --upgrade pip

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

RUN pip install --no-cache-dir -r requirements.txt

COPY src/BACKEND src/
COPY src/MODELS src/MODELS

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]