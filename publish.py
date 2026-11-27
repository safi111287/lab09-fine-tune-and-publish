"""Publish a trained model and optionally the public Gradio Space."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from share import write_model_card


def publish(model_id: str, space_id: str | None, model_dir: Path) -> None:
    if not all(
        (model_dir / name).is_file()
        for name in ("model.safetensors", "tokenizer.json", "config.json")
    ):
        raise SystemExit("Run the final experiment first to create the model files.")
    write_model_card(model_dir, Path("reports/final.json"))
    api = HfApi(token=True)
    api.create_repo(model_id, repo_type="model", private=False, exist_ok=True)
    if api.model_info(model_id).private:
        raise SystemExit("The Hugging Face model repository must be public.")
    api.upload_folder(
        repo_id=model_id,
        repo_type="model",
        folder_path=model_dir,
        allow_patterns=["*.json", "*.safetensors", "vocab.txt", "README.md"],
    )
    print(f"Model: https://huggingface.co/{model_id}")
    if space_id:
        api.create_repo(
            space_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        if api.space_info(space_id).private:
            raise SystemExit("The Hugging Face Space must be public.")
        api.add_space_variable(space_id, key="MODEL", value=model_id)
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path="space",
            allow_patterns=["app.py", "requirements.txt", "README.md"],
        )
        print(f"Space: https://huggingface.co/spaces/{space_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--space-id")
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    args = parser.parse_args()
    publish(args.model_id, args.space_id, args.model_dir)


if __name__ == "__main__":
    main()
