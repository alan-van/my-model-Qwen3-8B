import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download full Hugging Face model snapshot to local directory.")
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-8B", help="Model repo on Hugging Face (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--output_dir", type=str, default="./models/base/Qwen3-8B", help="Local directory to save model snapshot")
    args = parser.parse_args()

    print(f"Downloading snapshot: {args.repo_id} to {args.output_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete!") 