"""Run the complete assignment sequentially. Start with a GPU if available."""
import subprocess, sys
commands = [
    [sys.executable,"src/q1.py"],
    [sys.executable,"src/q2.py"],
    [sys.executable,"src/q3.py"],
    [sys.executable,"src/q4.py"],
]
for cmd in commands:
    print("\nRUNNING:", " ".join(cmd), flush=True)
    subprocess.run(cmd,check=True)
print("\nAll experiments completed. See ./outputs/")

