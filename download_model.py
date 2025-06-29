import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face model and tokenizer to local directory.")
    parser.add_argument("--model_repo", type=str, default="Qwen/Qwen3-8B", help="Model repo on Hugging Face (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--output_dir", type=str, default="./models/base/Qwen3-8B", help="Local directory to save model and tokenizer")
    args = parser.parse_args()

    print(f"Downloading model: {args.model_repo} to {args.output_dir}")
    model = AutoModelForCausalLM.from_pretrained(args.model_repo, cache_dir=args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_repo, cache_dir=args.output_dir)
    print("Download complete!") 