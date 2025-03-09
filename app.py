import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain.docstore.document import Document

# Title of the web app
st.title("📚 FAQ Search Engine with FAISS")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Try multiple encodings to handle special characters
    encodings = ["utf-8-sig", "latin1", "ISO-8859-1", "utf-16"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.success(f"CSV loaded successfully using encoding: {encoding}")
            break  # Stop once it works
        except Exception as e:
            st.warning(f"Failed with encoding {encoding}: {e}")

    if df is None:
        st.error("❌ Could not load CSV. Please check file encoding.")
        st.stop()

    # Normalize column names by stripping spaces and converting to lowercase
    df.columns = df.columns.str.strip().str.lower()

    if "output" not in df.columns:
        st.error(f"Error: CSV must have an 'output' column. Found columns: {list(df.columns)}")
        st.stop()

    # Convert FAQ data into LangChain Documents
    faq_texts = df['output'].dropna().tolist()

    if not faq_texts:
        st.error("Error: The 'output' column is empty!")
        st.stop()

    documents = [Document(page_content=str(row)) for row in faq_texts]

    # Initialize Hugging Face embeddings (No API key required)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector database
    vector_db = FAISS.from_documents(documents, embeddings)
    st.success("FAISS index built successfully! You can now search.")

    # Search bar for queries
    query = st.text_input("Ask a question:")
    if query:
        results = vector_db.similarity_search(query, k=3)  # Get top 3 matches
        st.subheader("Top Matches:")
        for i, result in enumerate(results):
            st.write(f"**{i+1}.** {result.page_content}")

# Footer
st.markdown("Made with ❤️ using Streamlit & FAISS")
