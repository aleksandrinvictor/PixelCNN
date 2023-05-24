import argparse
import torchvision
import numpy as np
from src.pixelcnn.infer import InferModel, INFER_CONFIGS

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "celeba", "cifar10"])
parser.add_argument("--num_images", type=int, default=144)
parser.add_argument("--device", type=str, default="cuda")
if __name__ == "__main__":
    args = parser.parse_args()

    infer_config = INFER_CONFIGS.get(args.dataset)
    if not infer_config:
        raise ValueError(f"Dataset {args.dataset} not found, available: {INFER_CONFIGS.keys()}")

    model = InferModel(infer_config=infer_config, checkpoint_path=args.checkpoint_path, device=args.device)

    sample = model.generate(num_images=args.num_images)

    nrow = int(np.sqrt(args.num_images))

    torchvision.utils.save_image(sample, f"sample_{args.dataset}_07.png", nrow=nrow, padding=0)
