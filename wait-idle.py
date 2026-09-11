"""Wait for three idle observations before changing the GPU serving workload."""
import subprocess
import time
import urllib.request

streak = 0
while streak < 3:
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=utilization.gpu,utilization.memory,power.limit',
        '--format=csv,noheader,nounits'], text=True)
    rows = [[float(x) for x in line.split(',')] for line in raw.splitlines()]
    assert all(row[2] <= 400 for row in rows), raw
    with urllib.request.urlopen('http://127.0.0.1:15015/metrics', timeout=5) as response:
        metrics = response.read().decode()
    counts = [float(line.rsplit(' ', 1)[1]) for line in metrics.splitlines()
              if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    idle = len(counts) >= 2 and all(x == 0 for x in counts)
    idle &= all(row[0] == row[1] == 0 for row in rows)
    streak = streak + 1 if idle else 0
    print('Idle observations:', streak, flush=True)
    if streak < 3:
        time.sleep(5)
