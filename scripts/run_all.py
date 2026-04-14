import subprocess

print('==START==')

token = input("Podaj HF token: ")
from huggingface_hub import login
login(token)

subprocess.run(["python3", "scripts/train_seq.py"], check=True)
subprocess.run(["python3", "scripts/evaluate_seq.py"], check=True)