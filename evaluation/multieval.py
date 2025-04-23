import subprocess
import torch
import torch.multiprocessing as mp
import datasets
import resource
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


def worker_main(device_id, queue):
    device = f"cuda:{device_id}"

    while not queue.empty():
        eval_args = queue.get()
        assert "press_name" in eval_args
        eval_args["device"] = device
        args = ["python", "evaluate.py"]

        for key, value in eval_args.items():
            args.append(f"--{key}={str(value)}")

        print("Executing:", *args)

        usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        retval = subprocess.run(args)
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)

        print(retval, f"{device}: took {usage_end.ru_utime - usage_start.ru_utime}")
        queue.task_done()


def setup():
    assert dataset_id in DATASET_DICT
    ds_builder = datasets.load_dataset_builder(DATASET_DICT[dataset_id], dataset_config)
    ds_builder.download_and_prepare()
    snapshot_download(model_repo, repo_type="model")


def main():
    manager = mp.Manager()
    queue = manager.Queue()

    for job in jobs:
        queue.put(dict(**global_settings, **job))

    for i in range(torch.cuda.device_count()):
        p = mp.Process(target=worker_main, args=(i, queue))
        p.start()

    queue.join()


if __name__ == "__main__":
    main()
