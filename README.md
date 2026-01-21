# span-finder

Parse sentences by finding & labeling spans.

## Requirements

- Python 3.14+
- PyTorch 2.9.1+

## Installation

### Install from PyPI (recommended)

```shell
pip install span-finder
```

### Install from source

```shell
# Clone the repository
git clone https://github.com/your-username/span-finder.git
cd span-finder

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Install with GPU support (conda)

```shell
conda create -n spanfinder python=3.14
conda activate spanfinder

# Install PyTorch with CUDA support first
pip install torch>=2.9.1 --index-url https://download.pytorch.org/whl/cu124

# Then install span-finder
pip install span-finder
```

## Quick Start

The simplest way to use SpanFinder is through the `load_model` function:

```python
from sftp import load_model

# Load model on CPU
predictor = load_model(device="cpu")

# Or load on GPU
predictor = load_model(device="cuda")

# Predict spans on a sentence
result = predictor.predict_sentence("Bob saw Alice eating an apple.")

# Print the parse tree
result.span.tree(result.sentence)
```

Output:
```
<Span Annotation: 3 descendents>
  <Span: (saw), Perception_active, 2 children>
    [Span: (Bob), Perceiver_agentive]
    [Span: (Alice eating an apple), Phenomenon]
  <Span: (eating), Ingestion, 2 children>
    [Span: (Alice), Ingestor]
    [Span: (an apple), Ingestibles]
```

## Usage Examples

### Single Sentence Prediction

```python
from sftp import load_model

# Load the model (downloads pretrained FrameNet model by default)
predictor = load_model(device="cpu")

# Predict on a string (will be tokenized automatically)
result = predictor.predict_sentence("The cat sat on the mat.")
result.span.tree(result.sentence)

# Predict on pre-tokenized input
tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
result = predictor.predict_sentence(tokens)
result.span.tree(result.sentence)
```

### Batch Prediction

```python
from sftp import load_model

predictor = load_model(device="cuda")

sentences = [
    "Bob saw Alice eating an apple.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
]

# Process multiple sentences efficiently
results = predictor.predict_batch_sentences(
    sentences,
    max_tokens=512,  # Maximum tokens per batch
    progress=True,   # Show progress bar
)

for result in results:
    print(f"Sentence: {' '.join(result.sentence)}")
    result.span.tree(result.sentence)
    print("-" * 40)
```

### Using a Custom Model

```python
from sftp import load_model

# Load from a local path
predictor = load_model(
    model_path="/path/to/your/model.tar.gz",
    device="cuda:0"
)

# Or from a URL
predictor = load_model(
    model_path="https://example.com/model.tar.gz",
    device="cpu"
)
```

### Accessing Prediction Results

```python
from sftp import load_model

predictor = load_model(device="cpu")
result = predictor.predict_sentence("Bob saw Alice eating an apple.")

# Access the span tree
span = result.span

# Get JSON representation
json_output = span.to_json()
print(json_output)

# Iterate over top-level spans (events/frames)
for event in span:
    print(f"Event: {event.label} at ({event.start_idx}, {event.end_idx})")

    # Iterate over arguments
    for arg in event:
        print(f"  Argument: {arg.label} at ({arg.start_idx}, {arg.end_idx})")
```

### Performance Optimization

```python
from sftp import load_model

predictor = load_model(device="cuda")

# Limit decoding depth and spans for faster inference
predictor.economize(
    max_decoding_spans=20,    # Limit number of predicted spans
    max_recursion_depth=2,    # Limit tree depth
)

result = predictor.predict_sentence("A complex sentence with many possible frames.")
```

## API Reference

### `load_model(model_path, device)`

Load a SpanFinder model for inference.

**Parameters:**
- `model_path` (str): Path to model checkpoint. Can be a URL, local `.tar.gz` file, or extracted directory. Default: FrameNet 1.7 pretrained model.
- `device` (str): Device to run on. Options: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"`, etc. Default: `"cpu"`.

**Returns:** `SpanPredictor` instance.

### `SpanPredictor.predict_sentence(sentence, output_format)`

Predict spans on a single sentence.

**Parameters:**
- `sentence` (str | list[str]): Input sentence as string or list of tokens.
- `output_format` (str): Output format - `"span"`, `"json"`, or `"concrete"`. Default: `"span"`.

**Returns:** `PredictionReturn` with `.span`, `.sentence`, and `.meta` attributes.

### `SpanPredictor.predict_batch_sentences(sentences, max_tokens, progress)`

Predict spans on multiple sentences efficiently.

**Parameters:**
- `sentences` (list): List of sentences.
- `max_tokens` (int): Maximum tokens per batch. Default: 512.
- `progress` (bool): Show progress bar. Default: False.

**Returns:** List of `PredictionReturn` objects.

## Command-Line Interface

SpanFinder also provides a CLI for quick predictions:

```shell
# Predict on a sentence
span-finder predict "Bob saw Alice eating an apple."

# Use GPU
span-finder predict "Bob saw Alice eating an apple." --device cuda

# Output as JSON
span-finder predict "Bob saw Alice eating an apple." --format json

# Use a custom model
span-finder predict "Bob saw Alice eating an apple." -m /path/to/model.tar.gz

# Read from stdin
echo "The cat sat on the mat." | span-finder predict

# Check version
span-finder --version
```

## Demo

A demo (combined with Patrick's coref model) is available at [nlp.jhu.edu/demos/lome](https://nlp.jhu.edu/demos/lome).

## Training

**Note:** Training requires AllenNLP, which is pinned to older PyTorch versions (< 1.13).
For training, you'll need a separate environment with compatible versions:

```shell
# Create a training environment with older PyTorch
pip install "span-finder[training]"

# Or install AllenNLP separately
pip install allennlp==2.10.1 allennlp-models==2.10.1
```

For training documentation, refer to:
- [Overall document](docs/overall.md)
- [Data format](docs/data.md)
- [Training guide](docs/training.md)

**Inference vs Training:**
- **Inference** (default): Works with PyTorch 2.9.1+ and Python 3.14+. Just `pip install span-finder`.
- **Training**: Requires AllenNLP with PyTorch < 1.13. Use `pip install "span-finder[training]"` in a compatible environment.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{xia-etal-2021-lome,
    title = "{LOME}: Large Ontology Multilingual Extraction",
    author = "Xia, Patrick  and
      Qin, Guanghui  and
      Vashishtha, Siddharth  and
      Chen, Yunmo  and
      Chen, Tongfei  and
      May, Chandler  and
      Harman, Craig  and
      Rawlins, Kyle  and
      White, Aaron Steven  and
      Van Durme, Benjamin",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    year = "2021",
    url = "https://www.aclweb.org/anthology/2021.eacl-demos.19",
    pages = "149--159",
}
```
