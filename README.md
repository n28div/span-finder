# span-finder

Hierarchical span finder and semantic role labeler trained on FrameNet 1.7.

Originally from [https://github.com/hiaoxui/span-finder](https://github.com/hiaoxui/span-finder). This fork adds HuggingFace Hub support so the model can be loaded directly with `from_pretrained`.

**Requires Python 3.8**.

## Installation

```shell
pip install git+https://github.com/n28div/span-finder.git
```

## Usage

```python
from sftp import SpanFinderHF

# load from HuggingFace Hub or a local directory
model = SpanFinderHF.from_pretrained("n28div/lome-spanfinder")

# single sentence (pre-tokenized)
result = model.predict(tokens=["John", "gave", "Mary", "a", "book", "."])

# batch
results = model.predict_batch(tokens=[
    ["John", "gave", "Mary", "a", "book", "."],
    ["She", "left", "the", "room", "."],
])
```

### Output format

Each call returns a dict (or list of dicts for batches):

```python
{
    "tokens": ["John", "gave", "Mary", "a", "book", "."],
    "frames": [
        {
            "name": "Giving",          # FrameNet frame label
            "idxs": [1, 1],            # [start, end] token indices of the trigger, inclusive
            "activation": "gave",      # trigger text
            "roles": [
                {
                    "name": "Donor",           # frame element label
                    "filler": "John",          # text span filling the role
                    "idxs": [0, 0]             # [start, end] token indices, inclusive
                },
                {
                    "name": "Recipient",
                    "filler": "Mary",
                    "idxs": [2, 2]
                },
                {
                    "name": "Theme",
                    "filler": "a book",
                    "idxs": [3, 4]
                }
            ]
        }
    ]
}
```

`frames` is a flat list — one entry per detected event in the sentence. Each frame has a trigger span (`idxs`, `activation`) and a list of role-filler pairs. All indices are token-level and inclusive on both ends.

By default predictions are filtered to FrameNet 1.7 frames and their valid frame elements. Pass `use_framenet=False` to get the raw model output instead.


## Citation

If you use this model for your research, please cite the original paper:

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
