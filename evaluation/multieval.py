import subprocess
import torch
import torch.multiprocessing as mp
import datasets
from huggingface_hub import snapshot_download
from evaluate import DATASET_DICT

dataset_id = "ruler"
dataset_config = "4096"
model_repo = "meta-llama/Meta-Llama-3.1-8B-Instruct"

global_settings = {
    "model": model_repo,
    "dataset": dataset_id,
    "data_dir": dataset_config,
    "press_name": "observed_attention",
}

jobs = [
    {"compression_ratio": 0.5},
    {"compression_ratio": 0.0, "quantization": 8},
    {"compression_ratio": 0.6},
    {"compression_ratio": 0.2, "quantization": 8},
    {"compression_ratio": 0.7},
    {"compression_ratio": 0.4, "quantization": 8},
    {"compression_ratio": 0.75},
    {"compression_ratio": 0.5, "quantization": 8},
]


def worker_eval(queue, eval_args):
    device = queue.get()
    assert "press_name" in eval_args

    eval_args["device"] = device

    args = ["python", "evaluate.py"]
    for key, value in eval_args.items():
        args.append(f"--{key}={str(value)}")

    print("Executing:", *args)

    retval = subprocess.run(args)
    print(retval)
    queue.put(device)


def setup():
    assert dataset_id in DATASET_DICT
    ds_builder = datasets.load_dataset_builder(DATASET_DICT[dataset_id], dataset_config)
    ds_builder.download_and_prepare()
    snapshot_download(model_repo, repo_type="model")


def main():
    manager = mp.Manager()
    gpus = torch.cuda.device_count()
    q = manager.Queue(maxsize=gpus)
    for i in range(gpus):
        q.put(f"cuda:{i}")

    processes = []
    for job in jobs:
        p = mp.Process(target=worker_eval, args=(q, dict(**global_settings, **job)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

