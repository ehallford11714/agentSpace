# agentSpace

Agentic framework for parsing agents.

## Streamlit PowerPoint Mapper

This repository includes a small Streamlit application (`streamlit_ppt_mapper.py`) that demonstrates how to populate a PowerPoint template with data from CSV files using an LLM.

### Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run streamlit_ppt_mapper.py
   ```
3. Provide your OpenAI API key, upload a PPTX template, a text file describing how the data should map into the slides, and one or more CSV files. The app will call the LLM to determine the mapping and return an updated presentation for download.

The LLM response should be a JSON array of actions with fields `slide_index`, `placeholder`, and `text`. The app applies these actions using `python-pptx`.
