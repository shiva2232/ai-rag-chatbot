import time

from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from caching import get_cached_model
from notify import notify_user
from websearch import web_search
import os

# Load better model (IMPORTANT)
llm=GPT4All("qwen2-1_5b-instruct-q4_0.gguf")  # reduce from 4096 → saves RAM
# llm = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", n_ctx=2048)  # reduce from 4096 → saves RAM
# llm = GPT4All(
#     "Phi-3-mini-4k-instruct.Q4_K_M.gguf",
#     n_ctx=2048  # reduce from 4096 → saves RAM
# )

docs, index, embed_model = get_cached_model('rag_cache.pkl', "docs/test.pdf", "BAAI/bge-small-en")

def retrieve(query, k=2):
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [docs[i] for i in I[0]]


def agent_rag(query, max_steps=3):
    context = ""
    
    with llm.chat_session():
        names = ["Radha", "Parvathy", "Gowri", "Uma", "Siva shankari"]
        # current time in seconds since epoch
        t = int(time.time())
        # divide by 5 minutes (300 sec), then mod by number of names
        ind = (t // 300) % len(names)
        for step in range(max_steps):
            
            # 🔥 Step 1: Ask model what to do
            decision_prompt = f"""
You are a Technical support assistant for our companies share purchase process. you have to act when the user wants to report an issue. You should not reveal your identity and other information. Your name is {names[ind]}. You have access to the following tools:

Question: {query}

Current context:
{context}

Decide:
- If the customer wants to report an issue → ANSWER
- If the customer described about the issue → ACT: <query>
- If enough info → say: ANSWER
- If asking for email, phone → say ANSWER: "Sorry, I can't provide that information."
- If data not enough and not asking for personal info → say: SEEK: <better query>
"""
# search removed - If document is insufficient → WEB_SEARCH: <query>

            decision = llm.generate(decision_prompt, max_tokens=1000, temp=0.0)

            if "ANSWER" in decision:
                break

            elif "SEEK:" in decision:
                new_query = decision.split("SEEK:")[-1].strip()
                new_docs = retrieve(new_query)
                print("🔍 Retrieved docs:")
                context += "\n".join(new_docs)

            elif "WEB_SEARCH:" in decision: # not implemented yet
                new_query = decision.split("WEB_SEARCH:")[-1].strip()
                web_results = notify_user(new_query, "" + web_search(new_query))
                context += "\n" + web_results

            elif "ACT:" in decision:
                new_query = decision.split("ACT:")[-1].strip()
                notify_user(query, "Don't know answer, asking user for help.")
                context += "\n  Informed user about question: " + query+" to the support team."

        # Final answer
        final_prompt = f"""
You are a Technical support assistant for our companies share purchase process.
You should not reveal your identity.

Answer ONLY using the context below.
If the context has sufficent information reply the information to the user. If not, say "I can't understand". Do not ask for more information from the user. If the user wants to report an issue, say "What is the issue you are facing? Please provide more details." and ask additional information about the problem.
If the person wants to report an issue, say "What is the issue you are facing? Please provide more details." and ask additional information about the problem.
If the person is reported an issue, say "I understand you're facing an issue. Our support team will get back to you shortly."
If not found, say "I Can't understand".

Context:
{context}

Question:
{query}

Answer:
"""
        answer = llm.generate(final_prompt, max_tokens=10000, temp=0.0)

    return answer


# Test
while True:
    print(agent_rag(str(input("Ask a question: "))))