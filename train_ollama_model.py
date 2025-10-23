import os
import subprocess
from pathlib import Path

def create_modelfile(base_model, docs_dir, output_modelfile):
    docs_content = ""
    for file_path in Path(docs_dir).glob("*"):
        if file_path.suffix in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                docs_content += f"\n\n# From {file_path.name}\n{f.read()}\n"

    modelfile_content = f"""
        FROM {base_model}

        SYSTEM """You are a helpful assistant trained with the following materials:
        {docs_content}
        Use this context to provide accurate, document-based responses."""
        """

    with open(output_modelfile, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"[+] Modelfile created at {output_modelfile}")


def train_and_reload_model(base_model, model_name, docs_dir):
    modelfile_path = f"./{model_name}_Modelfile"
    create_modelfile(base_model, docs_dir, modelfile_path)

    subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True)
    print(f"[+] New model '{model_name}' created from base '{base_model}'.")

    subprocess.run(["ollama", "run", model_name, "--prompt", "Summarize the training data."], check=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an Ollama model with documents.")
    parser.add_argument("--base", type=str, required=True, help="Base Ollama model (e.g. llama3)")
    parser.add_argument("--name", type=str, required=True, help="Name for the new model")
    parser.add_argument("--docs", type=str, required=True, help="Path to folder containing .txt or .md docs")

    args = parser.parse_args()

    train_and_reload_model(args.base, args.name, args.docs)
