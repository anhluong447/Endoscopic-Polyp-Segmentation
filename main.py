import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Endoscopic Polyp Segmentation Pipeline (ResUNet)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the ResUNet model")
    train_parser.add_argument(
        "--images-path",
        type=str,
        required=True,
        help="Path to training images folder (e.g., /kaggle/input/bkai-igh-neopolyp/train/train/)"
    )
    train_parser.add_argument(
        "--masks-path",
        type=str,
        required=True,
        help="Path to training ground truth masks folder (e.g., /kaggle/input/bkai-igh-neopolyp/train_gt/train_gt/)"
    )
    train_parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=12, help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--checkpoint-path", type=str, default="unet_model.pth", help="Path to save the best model weights")
    train_parser.add_argument("--pretrained-path", type=str, default=None, help="Path to load pretrained model checkpoint")
    train_parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    train_parser.add_argument("--wandb-project", type=str, default="PolypSegment", help="WandB project name")
    train_parser.add_argument("--wandb-key", type=str, default=None, help="WandB API key")
    train_parser.add_argument("--display-step", type=int, default=50, help="Interval of steps to print training progress")
    train_parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cuda, cpu)")

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run model inference on test dataset")
    inference_parser.add_argument(
        "--test-images-path",
        type=str,
        required=True,
        help="Path to test images folder (e.g., /kaggle/input/bkai-igh-neopolyp/test/test/)"
    )
    inference_parser.add_argument(
        "--model-path",
        type=str,
        default="unet_model.pth",
        help="Path to trained model weights checkpoint file"
    )
    inference_parser.add_argument(
        "--output-dir",
        type=str,
        default="predicted_masks",
        help="Directory where predicted masks will be saved"
    )
    inference_parser.add_argument(
        "--submission-path",
        type=str,
        default="output.csv",
        help="Path to output submission CSV file containing RLE strings"
    )
    inference_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    inference_parser.add_argument("--visualize-count", type=int, default=5, help="Number of predictions to plot in visualization image")
    inference_parser.add_argument("--visualize-path", type=str, default="visualization.png", help="Path to save visualization plot image")
    inference_parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cuda, cpu)")

    args = parser.parse_args()

    if args.command == "train":
        from src.train import train_model
        train_model(
            images_path=args.images_path,
            masks_path=args.masks_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_path=args.checkpoint_path,
            pretrained_path=args.pretrained_path,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_key=args.wandb_key,
            display_step=args.display_step,
            device=args.device
        )
    elif args.command == "inference":
        from src.inference import run_inference
        run_inference(
            test_images_path=args.test_images_path,
            model_path=args.model_path,
            output_dir=args.output_dir,
            submission_path=args.submission_path,
            batch_size=args.batch_size,
            visualize_count=args.visualize_count,
            visualize_path=args.visualize_path,
            device=args.device
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
