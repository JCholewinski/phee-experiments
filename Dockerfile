# baza z GPU (CUDA)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ustaw katalog roboczy
WORKDIR /workspace

# kopiuj requirements
COPY requirements.txt .

# instalacja paczek
RUN pip install --no-cache-dir -r requirements.txt

# kopiuj cały projekt
COPY . .

# ustaw PYTHONPATH
ENV PYTHONPATH=/workspace

# domyślna komenda (możesz zmienić)
CMD ["python", "scripts/run_all.py"]