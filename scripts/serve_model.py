#!/usr/bin/env python3
"""Serve various models with appropriate backends."""

import argparse
import subprocess
import sys


MODEL_CONFIGS = {
    "Qwen/Qwen2.5-14B-Instruct": {
        "command": [
            "vllm",
            "serve",
            "Qwen/Qwen2.5-14B-Instruct",
            "--tool-call-parser",
            "hermes",
            "--enable-auto-tool-choice",
            "--port",
            "8000",
        ]
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "command": [
            "vllm",
            "serve",
            "Qwen/Qwen2.5-32B-Instruct",
            "--tool-call-parser",
            "hermes",
            "--enable-auto-tool-choice",
            "--port",
            "8000",
        ]
    },
    "openai/gpt-oss-20b": {
        "command": [
            "uv",
            "run",
            "python",
            "-m",
            "gpt_oss.responses_api.serve",
            "--checkpoint",
            "openai/gpt-oss-20b",
            "--inference-backend",
            "vllm",
        ]
    },
    "openai/gpt-oss-120b": {
        "command": [
            "uv",
            "run",
            "python",
            "-m",
            "gpt_oss.responses_api.serve",
            "--checkpoint",
            "openai/gpt-oss-120b",
            "--inference-backend",
            "vllm",
        ]
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Serve various models with appropriate backends"
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to serve",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override default port (8000)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    # If no model specified or --list flag, show available models
    if args.list or args.model is None:
        print("Available models:")
        for model_name, config in MODEL_CONFIGS.items():
            print(f"  {model_name}")
        print("\nUsage: serve-model <model-name> [--port PORT]")
        sys.exit(0)

    config = MODEL_CONFIGS[args.model]
    command = config["command"].copy()

    # Override port if specified
    if args.port:
        if "--port" in command:
            port_idx = command.index("--port")
            command[port_idx + 1] = str(args.port)
        else:
            command.extend(["--port", str(args.port)])

    print(f"Starting {args.model}...")
    print(f"Command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running model: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)


if __name__ == "__main__":
    main()