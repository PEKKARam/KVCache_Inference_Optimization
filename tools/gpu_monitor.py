import argparse
import csv
import subprocess
import time

QUERY = "timestamp,utilization.gpu,memory.used,memory.total"


def read_nvidia_smi():
    cmd = [
        "nvidia-smi",
        f"--query-gpu={QUERY}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip().splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="CSV output path")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    rows = []
    start = time.time()
    while time.time() - start < args.duration:
        for line in read_nvidia_smi():
            rows.append(line.split(", "))
        time.sleep(args.interval)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gpu_util", "mem_used", "mem_total"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
