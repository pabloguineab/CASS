# RAG
Chainlit app for advanced RAG. Uses llamaparse, langchain, qdrant and OpenAI.

### create virtualenv
```
python3 -m venv .venv && source .venv/bin/activate
```

### Install packages
```
pip install -r requirements.txt
```

### Environment variables
All env variables goes to .env ( cp `example.env` to `.env` and paste required env variables)

### Run the python files (following the video to run step by step is recommended)
```
python3 ingest.py
chainlit run app.py
```

## Additional helper documents
- [LlamaIndex blogpost about Llamaparse](https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform)
- [Advanced demo with Reranker](https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb)
- [Parsing instructions Llamaparse](https://colab.research.google.com/drive/1dO2cwDCXjj9pS9yQDZ2vjg-0b5sRXQYo#scrollTo=dEX7Mv9V0UvM)

