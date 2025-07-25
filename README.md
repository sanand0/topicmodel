# TopicModel

TopicModel groups documents into clusters and names each cluster using OpenAI embeddings.
It generates a similarity matrix between documents and topics or discovers new topics when
a list is not provided.

## Installation

```bash
pip install topicmodel
```

Embeddings are cached in the file specified by the `TOPICMODEL_CACHE` environment
variable. By default it uses `~/.cache/topicmodel/embeddings.db` which works on
Windows, macOS and Linux.

## Usage

`uvx topicmodel --docs path/to/docs.csv`

### Document formats

**CSV**

```csv
text
Apples are great
Bananas are yellow
```

**JSON**

```json
[{"text": "Apples are great"}, {"text": "Bananas are yellow"}]
```

**TXT**

```text
Apples are great
Bananas are yellow
```

### Topics file

```csv
topic
Fruit
Vegetable
```

### Examples

Match documents to topics and save JSON:

```bash
uvx topicmodel --docs docs.csv --topics topics.csv --output result.json
```

Discover topics from a TXT file:

```bash
uvx topicmodel \
  --docs docs.txt \
  --output result.csv \
  --ntopics 5 \
  --name_model gpt-4.1-nano \
  --nsamples 3 \
  --truncate 100
```

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

```bash
uvx ruff --line-length 100 .
uvx --with pytest-asyncio,httpx,pandas,numpy,scikit-learn,tiktoken,tqdm pytest
```
