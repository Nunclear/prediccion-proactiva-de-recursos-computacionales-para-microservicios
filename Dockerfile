FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends git patch libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY ampliacion_20261110/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128 && pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /workspace
CMD ["bash"]
