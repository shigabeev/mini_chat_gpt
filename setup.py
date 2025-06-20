from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
requirements = []
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="mini_chat_gpt",
    version="0.1.0",
    author="Ilya Shigabeev",
    packages=find_packages(),
    python_requires=">=3.9.0",
    install_requires=requirements,
) 