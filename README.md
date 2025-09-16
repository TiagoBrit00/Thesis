# Thesis – AI-Based Measurement of Innovation Attractiveness

This repository contains the Python code developed for the thesis. The workflow is divided into 7 files, organized according to their processing order:

1. **`1_data_processing.py`** – All steps for processing the dataset and creating the resulting corpus.  
2. **`2_data_stats.py`** – Calculation of data statistics after processing.  
3. **`3_ground_truth.py`** – Statistics computed on the ground truth data.  
4. **`4_embeddings.py`** – Creation of vector embeddings for the processed text.  
5. **`5_agentic_rag.py`** – Agentic Retrieval-Augmented Generation (RAG) workflow.  
6. **`6_base_llm.py`** – Base Large Language Model workflow.  
7. **`7_results.py`** – Analysis and visualization of the results.  

---

## API Key Usage

Throughout the repository, some scripts require access to the OpenAI API. For **security and privacy reasons, the actual API key is not included in this repository**.  

The key is loaded using:

```python
from KEYS import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```
