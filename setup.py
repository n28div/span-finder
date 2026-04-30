from setuptools import setup, find_packages

setup(
    name="span-finder",
    version="0.0.2",
    author="Guanghui Qin",
    description="Hierarchical span finder and semantic role labeler built on AllenNLP",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "allennlp==2.9.3",
        "transformers>=4.18.0,<5",
        "torch>=1.11.0",
        "numpy>=1.21.0",
        "scipy",
        "tqdm",
        "cached-path>=1.1.0",
        "charset-normalizer",
        "huggingface_hub>=0.9.0",
    ],
)
