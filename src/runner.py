import argparse
import csv
import time
from pathlib import Path

import requests
import yaml


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts = load_prompts(cfg["prompts_file"])
    url = cfg["service_url"].rstrip("/") + cfg["endpoint"]
    out_path = Path(cfg["output_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    start_all = time.time()
    for i in range(cfg["requests"]):
        prompt = prompts[i % len(prompts)]
        payload = {
            "model": cfg["model"],
            "prompt": prompt,
            "max_tokens": cfg["max_tokens"],
            "temperature": cfg["temperature"],
        }
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=120)
        latency = time.time() - t0
        ok = 1 if resp.ok else 0
        rows.append([i, latency, ok, resp.status_code])

    total_time = time.time() - start_all
    rps = cfg["requests"] / total_time if total_time > 0 else 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "latency_s", "ok", "status_code"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["total_time_s", total_time])
        writer.writerow(["requests", cfg["requests"]])
        writer.writerow(["rps", rps])


if __name__ == "__main__":
    main()
