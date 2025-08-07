# topicmodel

`topicmodel` lets you discover what topics are covered in a bunch of documents. You can also classify documents into topics and find the similarity of each document with each topic.

## Usage

To categorize each line in `docs.txt` into topics, run:

```bash
export OPENAI_API_KEY=...
uvx topicmodel docs.txt --output topicmodel.txt
```

## Discover Topics

For example, if `docs.txt` has:

```text
Mars has a thin atmosphere.
The moon orbits Earth.
Stars shine at night.
Bread needs yeast.
Basil smells fresh.
```

Run:

```bash
uvx topicmodel docs.txt --ntopics=2
```

It groups each line into 2 _auto-discovered_ topics (e.g. "Celestial and Planetary Facts" and "Cooking and Aromatics") and print something like:

| text                        | best_match                    | Celestial and Planetary Facts | Cooking and Aromatics |
| --------------------------- | ----------------------------- | ----------------------------: | --------------------: |
| Mars has a thin atmosphere. | Celestial and Planetary Facts |                       0.29329 |               0.09759 |
| The moon orbits Earth.      | Celestial and Planetary Facts |                       0.33986 |               0.00704 |
| Stars shine at night.       | Celestial and Planetary Facts |                       0.34402 |               0.10109 |
| Bread needs yeast.          | Cooking and Aromatics         |                       0.04018 |               0.16585 |
| Basil smells fresh.         | Cooking and Aromatics         |                       0.05278 |               0.30279 |

The `best_match` column is the closest topic to the text. The rest of the columns are the similarity between the text and each topic.

## Use Existing Topics

Create this `topics.txt`:

```text
Astronomy
Cooking
```

Run:

```bash
uvx topicmodel docs.txt --topics topics.txt
```

This groups each line into the 2 topics in `topics.txt` along with the similarities:

| text                        | best_match | Astronomy | Cooking |
| --------------------------- | ---------- | --------: | ------: |
| Mars has a thin atmosphere. | Astronomy  |   0.17034 | 0.03036 |
| The moon orbits Earth.      | Astronomy  |   0.29521 | 0.01998 |
| Stars shine at night.       | Astronomy  |   0.28186 | 0.12287 |
| Bread needs yeast.          | Cooking    |   0.03838 | 0.18655 |
| Basil smells fresh.         | Cooking    |   0.05344 | 0.16860 |

## Options

- `--docs`: File containing documents. Required. Can be `.txt`, `.csv` or `.json`
  - `.txt`: Each line is treated as a document.
  - `.csv`: Each row is treated as a document. Only the first column is used.
  - `.json`: This should have an array of objects. Only the first key is used. Example: `[{"text": "Apples are great"}, {"text": "Bananas are yellow"}]`
- `--topics`: Optional file with existing topics you want to match with. Can be `.txt`, `.csv` or `.json`
- `--output`: Path to save results. Can be `.csv`, `.json` or `.txt`.
- `--model`: Default: `text-embedding-3-small`. OpenAI embedding model. Use `text-embedding-3-large` for higher quality.
- `--name_model`: Default: `gpt-4.1-mini`. Model to name clusters.
- `--ntopics`: Default: 20. Approx. number of topics to auto-discover. Increase for more granular clusters.
- `--nsamples`: Default: 5. Documents to show the naming model from each cluster. Higher values may improve topic names but increase cost.
- `--truncate`: Default: 200. Characters from each document to send to the naming model. Adjust based on document length; shorter saves tokens.
- `--prompt`: Prompt sent to the naming model. Modify to control naming style.

The default `--prompt` is:

> Here are clusters of documents. Suggest 2-4 word topic names for each cluster.
> Capture the spirit of each cluster. Differentiate from other clusters.

Environment variables:

```bash
# Use a different OpenAI compatible provider, e.g. openrouter:
export OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Embeddings are cached in this path. You can change it. The default is:
export TOPICMODEL_CACHE=~/.cache/topicmodel/embeddings.db
```

## Development

```bash
git clone https://github.com/gramener/topicmodel.git
cd topicmodel
uvx ruff --line-length 100 .
uvx --with pytest-asyncio,httpx,pandas,numpy,scikit-learn,tiktoken,tqdm pytest
```

## Deployment

Modify the `pyproject.toml` file to change the version number.

```bash
uv build
uv publish
```

This is deployed to [pypi](https://pypi.org/project/topicmodel/) as [Anand.S](https://pypi.org/user/Anand.S/)

## Change log

- [0.1.1](https://pypi.org/project/topicmodel/0.1.01): 07 Aug 2025. Help shows defaults. Informative errors. More tests
- [0.1.0](https://pypi.org/project/topicmodel/0.1.0/): 25 Jul 2025. Initial release

## License

[MIT](LICENSE)
