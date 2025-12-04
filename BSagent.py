# chatbot.py
from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
from typing import List, Tuple
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
from chunk_and_index import load_faiss_index, get_embedding
import faiss

client = OpenAI()


# ---------------------------------------------------------
# 1. PDF Search (FAISS)
# ---------------------------------------------------------
def search_pdf_index(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    index, metadata = load_faiss_index("index_data")
    if index is None:
        return []

    emb = get_embedding(query).astype(np.float32).reshape(1, -1)
    distances, idxs = index.search(emb, top_k)

    results = []
    for dist, idx in zip(distances[0], idxs[0]):
        if idx == -1:
            continue
        results.append((metadata[idx]["text"], float(dist)))

    return results


# ---------------------------------------------------------
# 2. Web Scraper (BeautifulSoup)
# ---------------------------------------------------------
def scrape_definition_from_web(query: str) -> str:
    query_clean = query.replace(" ", "+")

    urls = [
        f"https://indiankanoon.org/search/?formInput={query_clean}",
        f"https://www.latestlaws.com/search/?q={query_clean}",
    ]

    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        try:
            print(f"[DEBUG] Scraping: {url}")
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            # IndiaKanoon snippets
            snippets = soup.select(".snippet")
            if snippets:
                text = " ".join(s.get_text(" ", strip=True) for s in snippets)
                if len(text) > 80:
                    return text

            # Paragraphs
            paragraphs = soup.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
                if len(text) > 80:
                    return text

            # Div blocks
            divs = soup.find_all("div")
            text = " ".join(d.get_text(" ", strip=True) for d in divs)
            if len(text) > 80:
                return text

        except Exception as e:
            print("[DEBUG] Scraping error:", e)

    return ""


# ---------------------------------------------------------
# 3A. LLM Answer using PDF context (STRICT — PDF ONLY)
# ---------------------------------------------------------
def answer_from_pdf(query: str, context: str) -> str:
    prompt = f"""
You are a legal assistant. Use ONLY the text provided below to answer the user's question.
If the answer is present, extract or summarize it clearly.
If the answer is not present, respond with "NOT_FOUND".

QUESTION:
{query}

PDF CONTEXT:
{context}

Instructions:
- Answer concisely and accurately using only the PDF text.
- Do not guess or invent details.
- Extract the most relevant part of the text if it exists.
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return res.choices[0].message.content.strip()



# ---------------------------------------------------------
# 3B. LLM Final Answer using Web Text (FLEXIBLE)
# ---------------------------------------------------------
def answer_from_web(query: str, scraped_text: str) -> str:
    prompt = f"""
You are a legal assistant.

QUESTION:
{query}

You may use the scraped text for reference, but you CAN answer from
general legal knowledge if needed.

SCRAPED TEXT:
{scraped_text}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return res.choices[0].message.content.strip()


# ---------------------------------------------------------
# 4. Main QA Logic (PDF → Web Fallback)
# ---------------------------------------------------------
def ask_question(query: str) -> str:
    print("=== 🔍 Law Chatbot (PDF-first + Web Fallback) ===")

    # PDF-first logic
    pdf_hits_raw = search_pdf_index(query)
    answer = None

    if pdf_hits_raw:
        print(f"[DEBUG] {len(pdf_hits_raw)} PDF chunks found")
        for i, (text, dist) in enumerate(pdf_hits_raw):
            print(f"[DEBUG] Chunk {i} distance: {dist}")

        for chunk, _ in pdf_hits_raw:
            pdf_answer = answer_from_pdf(query, chunk)
            if pdf_answer != "NOT_FOUND":
                print("[DEBUG] Answer found in PDF chunk")
                answer = pdf_answer
                break

    if not answer:
        print("[DEBUG] Searching web...")
        scraped = scrape_definition_from_web(query)
        if scraped.strip():
            answer = answer_from_web(query, scraped)
        else:
            answer = "No information available."

    # Generate follow-up suggestions
    followups = suggest_followups(answer, query)
    if followups:
        answer += "\n\n💡 You might also ask:\n" + "\n".join(f"- " + q for q in followups)

    return answer


def suggest_followups(answer: str, query: str, max_suggestions: int = 3) -> list[str]:
    """
    Generate suggested follow-up questions based on the answer.
    """
    prompt = f"""
You are a legal assistant. The user just asked: "{query}".
You provided this answer:

{answer}

Suggest up to {max_suggestions} follow-up questions the user might ask next about this topic.
Provide each suggestion as a short, clear question.
Return only the questions in a list.
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text and split into lines
    suggestions_text = res.choices[0].message.content.strip()
    # Split by newlines or numbers/dashes
    suggestions = [line.strip("- ").strip() for line in suggestions_text.split("\n") if line.strip()]
    return suggestions[:max_suggestions]



# ---------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk your question: ")
        if q.lower() in ["exit", "quit"]:
            break

        print("\n--- ANSWER ---\n")
        print(ask_question(q))
