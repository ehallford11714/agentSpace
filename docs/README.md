# Semantic Pivot Demo

This repository includes a small demonstration for computing **semantic pivot** tokens from transformer embeddings. The demo uses a local implementation of robust PCA to separate an embedding matrix into low-rank and sparse components, then performs PCA to identify tokens that align strongly with the resulting components.

## Running the example

```
python examples/semantic_pivots.py --model gpt2 --k 64 --plot
```

- `--model` specifies any open-source model from the Hugging Face hub
- `--k` chooses how many principal components (pivot directions) to compute
- `--plot` visualizes the first two components with Matplotlib

Additional options:

- `--token TEXT_OR_ID` to project a specific token into the pivot space
- `--analogy A B C` to solve a simple A - B + C analogy in the pivot space

The script prints the tokens that are most aligned with each principal component. When plotting is enabled, the pivot tokens are labeled on a scatter plot of the first two components.

## Dependencies

The demo requires the following packages:

- `transformers`
- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`

Install them with `pip` before running the example.

