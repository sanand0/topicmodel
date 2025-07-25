# TopicModel

TopicModel groups documents into clusters and names each cluster using OpenAI embeddings.
It generates a similarity matrix between documents and topics or discovers new topics when
a list is not provided.

## Installation

```bash
pip install topicmodel
```

## Usage

`uvx topicmodel --docs path/to/docs.csv`

### Options

- `--docs`: CSV, JSON or TXT file containing documents. Required. Use CSV for structured data.
- `--topics`: Optional list of existing topics in CSV, JSON or TXT. Supply when you want to
  match documents against known categories.
- `--output`: Path to save results. Extension controls format: `.csv`, `.json` or `.txt`.
- `--model`: OpenAI embedding model. Defaults to `text-embedding-3-small`. Choose a larger
  model for better quality when documents are complex or short.
- `--ntopics`: Number of topics to discover when `--topics` is not supplied. Increase for
  more granular clusters.
- `--name_model`: Model used for naming clusters. Defaults to `gpt-4.1-nano`. Use a stronger
  model if names lack nuance.
- `--nsamples`: Documents to show the naming model from each cluster. Higher values may
  improve topic names but increase cost.
- `--truncate`: Characters from each document to send to the naming model. Adjust based on
  document length; shorter saves tokens.
- `--prompt`: Prompt sent to the naming model. Modify to control naming style.

## Development

Lint with `uvx ruff --line-length 100 .` and run tests with `pytest`.
