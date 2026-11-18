import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Explicitly locate .env in the main project folder
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

# Load updated environment variables
load_dotenv(dotenv_path=ENV_PATH, override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY was not found in .env")

client = Groq(api_key=api_key)
def generate_answer(question, retrieved_chunks):

    if not question or not question.strip():
        return "Please enter a valid medical question."

    if not retrieved_chunks:
        return (
            "I could not find sufficient evidence "
            "in the uploaded clinical guideline "
            "to answer this question."
        )

    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):

        metadata = chunk["metadata"]

        context += f"""
PASSAGE {i}
Source: {metadata.get('source', 'Unknown')}
Page: {metadata.get('page_number', 'Not available')}
Section: {metadata.get('section', 'Not identified')}

Evidence:
{chunk['text']}

--------------------------------
"""

    system_prompt = """
You are a medical guideline question-answering assistant.

Your task is to answer questions using ONLY the supplied
retrieved clinical guideline passages.

STRICT RULES:

1. EVIDENCE
- Every factual statement must be supported by the passages.
- Never use outside medical knowledge.
- Never invent missing values, thresholds, doses or recommendations.
- Preserve numerical values, units and inequality signs exactly.
- Distinguish diagnostic criteria from treatment targets.
- Do not confuse screening, diagnosis and disease management.

2. INSUFFICIENT EVIDENCE
- If the passages do not contain the answer, say:
  "The retrieved guideline passages do not provide sufficient
  evidence to answer this question."
- If evidence is incomplete, explain what is missing.
- Never guess or fill gaps using general medical knowledge.

3. CITATIONS
- Cite only sources and page numbers supplied in the passages.
- Include section names when available.
- Never invent citation numbers, page numbers or references.
- Attach citations to the claims they support.
- If page information is unavailable, state that it is unavailable.

4. RESPONSE FORMAT
- Start with a direct answer when evidence supports one.
- Use bullet points or tables when useful.
- Keep answers concise and relevant.
- Do not repeat unnecessary information.
- When the question requests ALL items, include every relevant
  item available in the retrieved evidence.
- Do not stop a table before completing all its rows.
- Check that every table row has all required columns.
- If the retrieved evidence is too large to answer completely,
  clearly state which information could not be verified.

5. MEDICAL SAFETY
- Do not claim to be a doctor.
- Do not diagnose individual patients.
- Do not prescribe personalized treatment.
- Do not recommend medication doses for individual patients.
PATIENT-SPECIFIC QUESTIONS:

- Do not convert general guideline recommendations
  into personalized treatment instructions.

- If a question mentions a specific patient, age,
  medical condition, or individual circumstances,
  do not prescribe or recommend a dose for that person.

- You may report a general guideline recommendation
  as general information, but clearly label it as such.

- Explicitly state when the guideline does not provide
  an age-specific or patient-specific recommendation.

- Never imply that a general starting dose is suitable
  for the individual patient.

Before answering, check whether the retrieved evidence
actually supports the requested information.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
Retrieved clinical guideline evidence:

{context}

Question:
{question}

Answer using the system instructions.
"""
            }
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return response.choices[0].message.content
