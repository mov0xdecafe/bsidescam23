import argparse

from dataset import FunctionIdentificationDataset

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="Path to directory contain binaries to include in dataset")

    args = arg_parser.parse_args()

    kernel_size = 20

    print("Function Boundary Identification")
    dataset = FunctionIdentificationDataset(args.dataset_path, block_size=1000, padding_amount=kernel_size-1)