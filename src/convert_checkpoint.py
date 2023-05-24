import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)
if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)

    model_state_dict = {}

    for k, v in checkpoint["state_dict"].items():
        model_state_dict[".".join(k.split(".")[1:])] = v

    torch.save({"state_dict": model_state_dict}, f"{os.path.splitext(args.checkpoint_path)[0]}.pt")
