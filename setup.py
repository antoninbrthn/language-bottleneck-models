from pathlib import Path
from setuptools import setup, find_packages


README_PATH = Path(__file__).parent / "README.md"

setup(
    name="language-bottleneck-models",
    version="0.1.0",
    description="Language Bottleneck Models for Qualitative Knowledge State Modeling",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "openai",
        "aiohttp",
        "hydra-core",
        "omegaconf",
        "wandb",
        "pydantic",
        "python-dotenv",
        "PyYAML",
        "tqdm",
        "tiktoken",
    ],
    extras_require={
        "unsloth": ["unsloth"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
