from setuptools import setup, find_packages


setup(
    name='sftp',
    version='1.0.0',
    author='Guanghui Qin',
    description='SpanFinder: Parse sentences by finding & labeling spans',
    packages=find_packages(),
    python_requires='>=3.14',
    install_requires=[
        'torch>=2.9.1',
        'transformers>=4.45.0',
        'numpy>=2.0.0',
        'allennlp>=2.10.1',
        'allennlp-models>=2.10.1',
        'spacy>=3.8.0',
        'tqdm',
    ],
)
