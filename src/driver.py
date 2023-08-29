import argparse

from dataset import FunctionIdentificationDataset

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="Path to directory contain binaries to include in dataset")

    args = arg_parser.parse_args()

    print("Function Boundary Identification")
    dataset = FunctionIdentificationDataset(args.dataset_path)