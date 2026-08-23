FROM python:3.11-slim

WORKDIR /app

# Python buffers stdout when it's not attached to a TTY (true for any container),
# so without this, print()/logging output sits invisible in a buffer until it fills —
# making a slow-but-healthy startup look like it produced no logs at all.
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=100

# sentence-transformers pulls in torch>=1.11.0 unconstrained, which resolves to the
# default PyPI wheel bundling CUDA/cuDNN (over 1GB) even though this app only runs a
# small embedding model on CPU. Installing the CPU-only build first satisfies that
# constraint before requirements.txt is processed, so pip never reaches for the GPU
# build — this is the largest single win for build time and image size.
RUN pip install --no-cache-dir --retries 10 --timeout 100 \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir --retries 10 --timeout 100 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]