# agentSpace
agentic framework for parsing agents

## Examples

- Install the minimal dependencies for the examples with
`pip install pandas numpy pydantic`.

- **Document analysis placeholder**: `python examples/document_analysis_workflow.py`
- **Data cleansing workflow**: `python examples/data_cleansing_workflow.py` to see a
  scan-plan-act-audit loop that cleans a toy dataframe using simple cooperative
  agents.
- **Hugging Face dataset agent**: `python examples/hf_dataset_agent.py` now
  defaults to the public `buio/heart-disease` dataset so you can see an
  end-to-end retrieval → EDA → cleaning → AutoML model search pass with
  lightweight hyperparameter sweeps, an auto-ensemble voting candidate, a
  leaderboard, and a held-out accuracy printout. You can still provide a
  different dataset, config, and split via CLI args (e.g., `python
  examples/hf_dataset_agent.py ag_news None train`). Pass `--save-dir
  runs/latest` to persist `pivot.json`, `data_sample.csv`, and
  `leaderboard.csv` for post-run review. Requires `pip install datasets
  scikit-learn`.
- **Pivot embeddings + HellaSwag comparison**: `python
  examples/pivot_embeddings_hellaswag.py` computes pivot embeddings with a
  small encoder, walks through a reasoning chain that realigns to the pivot, and
  then runs both a control HellaSwag accuracy pass (causal LM log-likelihood) and
  a pivot-embedding cosine scorer for side-by-side accuracy deltas. Use
  `--eval-samples 0` for the full validation set or a smaller integer for a quick
  subset. Requires `pip install datasets transformers torch sentence-transformers`.
- **Module entrypoint**: `python __main__.py` runs the Hugging Face dataset agent by
  default.
