Medical RAG Chatbot  Response Generation Experiment
Overview
This project explores improving response generation in a Medical
Retrieval-Augmented Generation (RAG) chatbot. The experiment uses the
WHO document HEARTS D: Diagnosis and Management of Type 2 Diabetes
(WHO-UCN-NCD-20.1, 2020).

The application extracts clinical document text, creates chunks,
generates embeddings, stores them in ChromaDB, retrieves relevant
passages and generates an answer using a Groq-hosted language model.
Main features
- Upload PDF, DOCX, TXT and HTML documents.
- Extract text and available source metadata.
- Configure chunk size and overlap.
- Generate normalized 384-dimensional embeddings with
  all-MiniLM-L6-v2.
- Store and retrieve document chunks with ChromaDB cosine similarity.
- Generate evidence-grounded answers using openai/gpt-oss-20b
  through Groq.
- Display retrieved passages and generated answers in Streamlit.
- Include source/page references when the retrieved metadata provides
  them.
Architecture
1. Document processing: extract text from the uploaded file.
2. Chunking: split text into overlapping character-based chunks.
3. Embedding: encode chunks using Sentence Transformers.
4. Vector storage: store embeddings, text and metadata in ChromaDB.
5. Retrieval: retrieve relevant passages for the user's question.
6. Response generation: pass the retrieved evidence to the LLM with
   evidence-control instructions.
Project structure
response_generation_experiment/
├── app.py
├── requirements.txt
├── data/
│   └── WHO-UCN-NCD-20.1-eng.pdf
├── prompts/
├── evaluation/
├── results/
└── src/
    ├── chatbot.py
    ├── chunking.py
    ├── document_processor.py
    ├── embeddings.py
    ├── retrieval.py
    └── vector_store.py
Setup
Create and activate a virtual environment:
python -m venv venv
.\venv\Scripts\Activate.ps1
Install dependencies:
pip install -r requirements.txt
Create a .env file in the project directory and add your Groq API key:
GROQ_API_KEY=your_groq_api_key
Do not commit .env or expose your API key. Add .env, venv/,
chroma_db/ and __pycache__/ to .gitignore.
Run the application
From the experiment directory:
streamlit run app.py
Upload a supported clinical document, process it, and then enter a
question in the question-answer section.
Response-generation improvements
The improved system prompt instructs the model to: - Answer using only
retrieved guideline passages. - Avoid unsupported facts, doses and
recommendations. - Preserve numerical values, units and inequality
signs. - Distinguish diagnostic criteria from management targets. - Cite
only source and page information present in the retrieved metadata. -
State when the retrieved evidence is insufficient.
Observed experiment results
- The final diagnostic-criteria test returned all five measurements
  and cut-off values from Table 3 of the WHO guideline.
- Earlier answers omitted some criteria, demonstrating that
  retrieval/context completeness matters as well as prompt design.
- A patient-specific insulin-dose question produced a general starting
  dose. This revealed a safety limitation: general guideline
  information must not be presented as an individualized prescription.
No aggregate benchmark score is reported because a complete scored
evaluation was not performed.
Limitations
- Retrieval quality depends on chunk boundaries, embedding similarity
  and retrieval settings.
- Page-level metadata does not guarantee precise section-level
  citations.
- Citation correctness requires additional automated or manual
  validation.
- The patient-specific treatment test exposed a medical-safety
  weakness.
- This project is for learning and document-based information
  retrieval, not diagnosis or treatment. It must not be used as a
  clinical decision-support system.
Future improvements
- Add stronger abstention rules for patient-specific medical
  questions.
- Validate generated citations against retrieved metadata.
- Add a fixed benchmark set and record manual scores consistently.
- Improve table-aware chunking and section metadata extraction.
- Test every supported file format with representative documents.
Technology stack
Python, Streamlit, PyMuPDF, python-docx, BeautifulSoup, Sentence
Transformers, ChromaDB, Groq API and openai/gpt-oss-20b.
