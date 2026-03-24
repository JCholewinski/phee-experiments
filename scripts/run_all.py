import subprocess

print('==START==')

token = input("Podaj HF token: ")
from huggingface_hub import login
login(token)

subprocess.run(["python", "scripts/train_seq.py"], check=True)
subprocess.run(["python", "scripts/evaluate_seq.py"], check=True)