"""sarthLLM — a GPT built from scratch, pretrained on WikiText-103.

The importable package is still `llm_from_scratch` (it matches the repo and the
from-scratch study layout); "sarthLLM" is the model and the CLI name:

    sarthllm "The capital of France is"
"""

__version__ = "0.1.0"


def main() -> None:
    """Console-script entry point. The real CLI lives in `llm_from_scratch.cli`
    and is registered as `sarthllm` in pyproject.toml; this is only the
    fallback for the `llm-from-scratch` script."""
    from llm_from_scratch.cli import main as cli_main

    raise SystemExit(cli_main())

