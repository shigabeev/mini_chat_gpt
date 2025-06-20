"""
Setup script for mini_chat_gpt package.
A clean, minimal implementation of GPT-2 for pretraining on TinyStories.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle commented-out requirements
                if line.startswith("# "):
                    continue
                requirements.append(line)

# Development dependencies for testing and linting
dev_requirements = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

setup(
    name="mini_chat_gpt",
    version="0.1.0",
    author="Mini ChatGPT Team",
    author_email="",
    description="A clean, minimal implementation of GPT-2 for pretraining on TinyStories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mini_chat_gpt",  # Update with actual URL
    packages=find_packages(exclude=["tests*", "checkpoints*", "outputs*", "wandb*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "flash": ["flash-attn"],  # Optional FlashAttention dependency
    },
    entry_points={
        "console_scripts": [
            "mini-chat-gpt-train=mini_chat_gpt.train:main",
            "mini-chat-gpt-generate=mini_chat_gpt.generate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mini_chat_gpt": ["config.yaml"],
    },
    zip_safe=False,
    keywords="gpt transformer language-model pytorch tinystories",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mini_chat_gpt/issues",
        "Source": "https://github.com/yourusername/mini_chat_gpt",
        "Documentation": "https://github.com/yourusername/mini_chat_gpt/blob/main/README.md",
    },
) 