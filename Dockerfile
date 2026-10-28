# GPU image for the training / evaluation environment (the LLM part uses requirements-llm.txt).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/workspace
CMD ["python", "scripts/run_matrix.py", "--groups", "baselines", "seq", "eqa", "gqa"]
