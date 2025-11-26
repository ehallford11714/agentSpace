# agentSpace
agentic framework for parsing agents

## Examples

- Install the minimal dependencies for the examples with
`pip install pandas numpy pydantic`.

- **Document analysis placeholder**: `python examples/document_analysis_workflow.py`
- **Data cleansing workflow**: `python examples/data_cleansing_workflow.py` to see a
  scan-plan-act-audit loop that cleans a toy dataframe using simple cooperative
  agents.
- **Hugging Face dataset agent**: `python examples/hf_dataset_agent.py` to fetch a
  public dataset (e.g., `ag_news` or `imdb`) using a STELLA-inspired loop that
  scouts metadata, plans, executes, audits a download, runs EDA, cleans the
  dataframe, and then trains a simple baseline model with a train/test split.
  Requires `pip install datasets scikit-learn`.
- **Module entrypoint**: `python __main__.py` runs the Hugging Face dataset agent by
  default.
