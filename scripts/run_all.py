import subprocess

print('==START==')
subprocess.run(["python", "scripts/train_seq.py"], check=True)
subprocess.run(["python", "scripts/evaluate_seq.py"], check=True)