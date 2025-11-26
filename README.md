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
  end-to-end retrieval → EDA → cleaning → model-building pass with an accuracy
  printout on the held-out set. You can still provide a different dataset,
  config, and split via CLI args (e.g., `python examples/hf_dataset_agent.py
  ag_news None train`). Requires `pip install datasets scikit-learn`.
- **Module entrypoint**: `python __main__.py` runs the Hugging Face dataset agent by
  default.
