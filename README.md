# receipts_reader

Read receipt images, extract information and output to csv file

# Features
- Uses multimodal LLM (structured output) to extract information from the images
- Information extracted include:
  - shop name
  - transaction date
  - net amount
- Output all information into a csv file

# Instructions
- Install Ollama (https://ollama.com/download/linux)
- Pull multimodal LLM (ollama pull mistral-small3.2:24b)
- Install libraries (pip install -r requirements.txt)
- Place receipt images under /images folder
- Run python file (python receipts_reader.py)

# Warning
- The specified LLM model requires high (GPU/CPU) memory.
