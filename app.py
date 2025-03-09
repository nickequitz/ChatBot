import streamlit as st
import pandas as pd
import requests
import re
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# ✅ Load Groq API Key (from Streamlit Secrets)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.warning("⚠️ No Groq API Key found. Set 'GROQ_API_KEY' in .streamlit/secrets.toml.")

# ✅ Define Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ Streamlit UI
st.title("💬 The Equitz AI-Powered FAQ Chatbot for inquiring University of Houston freshman 💬")
st.write("Ask a question, and we'll check the FAQ database first. If no match is found, Groq AI will assist.")

# ✅ Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your FAQ CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="utf-8")

    # ✅ Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # ✅ Check if "output" column exists
    if "output" not in df.columns:
        st.error(f"❌ Error: CSV must contain an 'output' column. Found columns: {list(df.columns)}")
        st.stop()

    st.success("✅ FAQ CSV Loaded Successfully!")

    # ✅ Convert FAQ data into LangChain Documents
    documents = [Document(page_content=str(row)) for row in df['output'].dropna().tolist()]

    # ✅ Initialize FAISS Vector Database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embeddings)
    st.success("✅ FAISS Vector Search Initialized!")

    # ✅ Function to Smart-Truncate Text at a Full Sentence
    def smart_truncate(text, min_length=200, max_length=1000):
        """
        Truncates text at the nearest full sentence before reaching max_length.
        Ensures a minimum length to retain meaningful context.
        """
        if len(text) <= max_length:
            return text  # No truncation needed

        sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by punctuation (., !, ?)
        truncated_text = ""

        for sentence in sentences:
            if len(truncated_text) + len(sentence) > max_length:
                break  # Stop before exceeding max_length
            truncated_text += sentence + " "

        return truncated_text.strip() if len(truncated_text) >= min_length else text

    # ✅ Function to Query FAISS
    def search_faiss(query, similarity_threshold=0.4, max_length=1000):
        results = vector_db.similarity_search_with_score(query, k=5)

        if results:
            best_answer, raw_score = results[0]
            similarity_score = 1 / (1 + raw_score)  # Convert L2 distance to similarity

            if similarity_score >= similarity_threshold:
                return smart_truncate(best_answer.page_content, max_length=max_length)

        return None  # No relevant match found

    # ✅ Function to Query Groq API
    def ask_groq(question, max_tokens=512, max_length=1000):
        """
        Sends a query to Groq AI and returns a detailed response.
        """
        if not GROQ_API_KEY:
            return "❌ Groq API Key missing. Set it in Streamlit Secrets."

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are a knowledgeable assistant that provides detailed, well-structured, and informative responses in a natural way."},
                {"role": "user", "content": question}
            ],
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data)

            if response.status_code == 200:
                raw_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No valid response from Groq API.")
                return smart_truncate(raw_response, max_length=max_length)
            else:
                return f"❌ Groq API Error {response.status_code}: {response.text}"

        except requests.exceptions.RequestException as e:
            return f"❌ API Request failed: {e}"

    # ✅ Chat Interface
    user_input = st.text_input("💬 Ask a question:")

    if user_input:
        faiss_answer = search_faiss(user_input)

        if faiss_answer:
            st.subheader("✅ Answer from UH FAQ Database:")
            st.write(faiss_answer)
        else:
            st.subheader("🤖 Groq AI Response, powered by Equitz:")
            groq_answer = ask_groq(user_input)
            st.write(groq_answer)
