import argparse

import mii


def main(args: argparse.Namespace):
    mii.serve(args.model)
    while True:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)
