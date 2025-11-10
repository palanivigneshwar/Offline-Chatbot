# Medical Chatbot

**AI-Powered Medical Chatbot for Patient Understanding**

## Installation (Internet Required)

1. **Install dependencies**
   Make sure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the LLM model**
   This step downloads the local language model used by the chatbot:

   ```bash
   python downloadModel.py
   ```

3. **Download the Embedding model**
   This step downloads the local language model used by the chatbot:

   ```bash
   python downloadEmbeddings.py
   ```

## Usage

Run the application using:

```bash
python demo.py
```

This will start a local Gradio interface at:
[http://127.0.0.1:7860/](http://127.0.0.1:7860/)

* Log in using the following credentials:
  **Username:** `user`
  **Password:** `12345`

* Upload a medical report or lab PDF.

* Ask the chatbot questions related to the uploaded file.

> **Note:**
> The chatbot uses a local LLM, so response times may be slower than cloud-based solutions.
> Ensure C++ is installed since LlamaCPP requires C++.


