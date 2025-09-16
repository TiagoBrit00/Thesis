import os
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

# Load API key from another file for security
from KEYS import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Working directories and parameters
pickle_path = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\all_documents.pkl"
faiss_index_path = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\faiss_index"
embedding_model = "text-embedding-3-small"
batch_size = 100  # Number of documents per batch

# Main processing function
def main():

    # Load documents from pickle file
    with open(pickle_path, "rb") as f:
        all_documents = pickle.load(f)

    # Number of documents loaded
    print(f"Loaded {len(all_documents)} documents.")
    
    # Sample document info (optional, for logging)
    sample_doc = all_documents[0]
    print(f"Sample content length: {len(sample_doc.page_content)} characters")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Calculate total batches
    total_batches = (len(all_documents) + batch_size - 1) // batch_size
    
    # Process documents in batches and build FAISS index
    faiss_index = None

    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        batch_index = FAISS.from_documents(batch, embeddings)

        if faiss_index is None:
            faiss_index = batch_index
        else:
            faiss_index.merge_from(batch_index)

    # Save the FAISS index
    faiss_index.save_local(faiss_index_path)
    print(f"Total vectors saved in FAISS index: {faiss_index.index.ntotal}")

# Run the main function
main()