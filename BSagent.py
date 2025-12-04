# -------------------------
# IMPORTS + ENV
# -------------------------
from dotenv import load_dotenv
load_dotenv()

import os
import requests
import pickle
import numpy as np
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_DIR = "index_data"

# -------------------------
# 1. LEGAL QUESTION CHECK
# -------------------------
LEGAL_KEYWORDS = [
    "law","legal","act","section","article","ipc","crpc",
    "right","penalty","offence","court","case","rights",
    "constitution","government","regulation","crime","justice"
]

def is_legal_question(question: str) -> bool:
    """Return True if question is legal."""
    if any(k in question.lower() for k in LEGAL_KEYWORDS):
        return True
    
    prompt = f"Answer YES or NO only. Is this a legal question?\nQuestion: {question}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip().lower().startswith("y")


# -------------------------
# 2. PDF SEARCH (FAISS)
# -------------------------
def load_faiss_index():
    import faiss
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path  = os.path.join(INDEX_DIR, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None
    
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search_pdf(question: str, top_k=5, threshold=0.6):
    index, metadata = load_faiss_index()
    if index is None:
        return None
    
    from chunk_and_index import get_embedding
    q_emb = np.array([get_embedding(question)], dtype=np.float32)
    
    distances, indices = index.search(q_emb, top_k)
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold and 0 <= idx < len(metadata):
            results.append(metadata[idx]["text"])

    return "\n".join(results) if results else None


# -------------------------
# 3. KEYWORD EXTRACTION
# -------------------------
def extract_keyword(question: str):
    prompt = f"Extract ONE keyword from this legal question:\n{question}\nKeyword:"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip().replace(" ", "_")


# -------------------------
# 4. SCRAPE TRUSTED SOURCES
# -------------------------
TRUSTED_DOMAINS = [
    "incometaxindia.gov.in",
    "cbic.gov.in",
    "mca.gov.in",
    "rbi.org.in",
    "indiankanoon.org",
    "ibbi.gov.in",
    "indiacode.nic.in",
    "legislative.gov.in"
]

def scrape_trusted_sources(keyword: str):
    print(f"[DEBUG] Searching trusted legal sites for: {keyword}")

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }

    SEARCH_URLS = [
        # IndiaCode search
        f"https://www.indiacode.nic.in/search?searchtype=1&text={keyword}",

        # Legislative.gov search
        f"https://www.legislative.gov.in/search/site/{keyword}",

        # IndiaKanoon direct search
        f"https://indiankanoon.org/search/?formInput={keyword}",

        # IPC (Old Code) – definitions
        f"https://indiacode.nic.in/show-data?actid=AC_CEN_5_23_00045_186045_1517807325716&sectionId=95&sectionno=1&orderno=1",

        # CrPC
        f"https://indiacode.nic.in/handle/123456789/2263?view_type=browse&sam_handle=123456789/1362"
    ]

    for url in SEARCH_URLS:
        print(f"[DEBUG] Trying: {url}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)

            if r.status_code not in [200, 302]:
                print(f"[DEBUG] Status {r.status_code} for {url}")
                continue

            soup = BeautifulSoup(r.content, "html.parser")

            # Extract large paragraphs
            paras = soup.find_all("p")
            text = "\n".join(
                p.get_text().strip()
                for p in paras
                if len(p.get_text().strip()) > 40
            )

            if len(text) > 150:
                print("[DEBUG] Extracted content from:", url)
                return text

        except Exception as e:
            print(f"[DEBUG] Error fetching {url}: {e}")

    print("[DEBUG] No trusted source returned text.")
    return None



# -------------------------
# 5. LLM ANSWER
# -------------------------
def answer_with_llm(context: str, question: str):
    prompt = f"Use ONLY the text below.\nQUESTION: {question}\nTEXT:\n{context}\nAnswer:"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()


# -------------------------
# 6. MAIN AGENT
# -------------------------
def legal_agent(question: str):
    if not is_legal_question(question):
        return "I can only answer legal questions."

    pdf_context = search_pdf(question)
    if pdf_context:
        return answer_with_llm(pdf_context, question)

    keyword = extract_keyword(question)
    web_context = scrape_trusted_sources(keyword)
    if web_context:
        return answer_with_llm(web_context, question)

    return "Information not available."


# -------------------------
# 7. CLI LOOP
# -------------------------
if __name__ == "__main__":
    print("\n=== 🔍 Law Chatbot (Trusted sources + PDF-first) Ready ===\n")
    while True:
        q = input("Ask your question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        
        print("\n--- ANSWER ---\n")
        print(legal_agent(q))
        print("\n----------------\n")
