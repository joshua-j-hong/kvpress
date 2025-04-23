import argparse
import ast
import multiprocessing

# import torch
import itertools
from .evaluate import PRESS_DICT, evaluate

parser = argparse.ArgumentParser(
    description="Run evaluations in parallel",
)

parser.add_argument(
    "--dataset",
    action="store",
    nargs=1,
    help=f"Dataset to evaluate",
)

parser.add_argument(
    "--data_dir",
    action="store",
    nargs=1,
    help=f"Dataset to evaluate",
)

parser.add_argument(
    "-p",
    "--press",
    action="store",
    nargs=1,
    help=f"Supported arguments: ${', '.join(PRESS_DICT.keys())}",
)

parser.add_argument(
    "-m",
    "--model",
    action="store",
    default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    nargs=1,
    help="Name of model to use",
)

parser.add_argument(
    "-r",
    "--ratios",
    action="store",
    help='Overall compression ratios to evaluate with. Use a Python-style list, e.g. --ratios "[0.5, 0.75]" ',
    type=ast.literal_eval,
)

parser.add_argument(
    "-t",
    "--dtype",
    action="store",
    help="Number of bits to load the model in",
    type=int,
    default=16,
)

parser.add_argument(
    "-q",
    "--quanto",
    action="store",
    help='Quantizations to try. Use a Python-style list, e.g. --quanto "[4, 8]"',
    type=ast.literal_eval,
    default=[],
)


def run_evaluation(arg):
    device_num = multiprocessing.current_process()._identity[0]
    print(device_num)

    pass


def execute():
    args = parser.parse_args()
    device_count = torch.cuda.device_count()

    press_ratios = []

    args.quanto.append(None)

    def compute_compression_ratio(
        arg: tuple[int | None, float],
    ) -> tuple[int | None, float]:
        quanto_bits, target_ratio = arg
        quant_ratio = 1 if nbits is None else (nbits / args.dtype)
        press_ratio = 1 - (1 - target_ratio) / quant_ratio
        return quanto_bits, press_ratio

    jobs = map(compute_compression_ratio, itertools.product(args.quanto, args.ratios))

    jobs = filter(lambda arg: arg[1] >= 0.0, jobs)

    def create_evaluation_args(arg):
        (quanto_bits, press_ratio) = arg
        return {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "model": args.model,
            "press_name": args.press,
            "compression_ratio": press_ratio,
            "quanto_bits": quanto_bits,
        }

    jobs = map(create_evaluation_args, jobs)

    with multiprocessing.Pool(device_count) as pool:
        pool.map(run_evaluation, jobs)


if __name__ == "__main__":
    execute()
