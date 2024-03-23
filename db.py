from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from chromadb.utils import embedding_functions

class TextChunkProcessor:
    def __init__(self, data_dir, db_path):
        self.data_dir = data_dir
        self.db_path = db_path
        self.documents = None
        self.text_parser = None
        self.db = None
        self.vector_store = None

    def load_documents(self):
        reader = SimpleDirectoryReader(self.data_dir)
        self.documents = reader.load_data()

    def initialize_text_parser(self, chunk_size=1024):
        self.text_parser = SentenceSplitter(chunk_size=chunk_size)

    def initialize_database(self, collection_name="Articles"):
        self.db = chromadb.PersistentClient(path=self.db_path)
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.vector_store = self.db.get_or_create_collection(name=collection_name, embedding_function=default_ef)

    def process_documents(self):
        if self.documents is None or self.text_parser is None or self.vector_store is None:
            raise ValueError("Documents, text parser, or vector store not initialized.")

        text_chunks = []
        doc_idxs = []
        doc_metadata = []
        chunk_ids = []

        for doc_idx, doc in enumerate(self.documents):
            cur_text_chunks = self.text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)

            # Store metadata and IDs for each text chunk
            for chunk_idx, _ in enumerate(cur_text_chunks):
                doc_idxs.append(doc_idx)
                doc_metadata.append(doc.metadata)
                chunk_ids.append(f"{doc_idx}_{chunk_idx}")  # Generating a unique ID for each chunk

        self.vector_store.add(ids=chunk_ids, documents=text_chunks, metadatas=doc_metadata)

def main():
    data_dir = "./data"
    db_path = "./chroma_db"
    
    processor = TextChunkProcessor(data_dir, db_path)
    processor.load_documents()
    processor.initialize_text_parser()
    processor.initialize_database()
    processor.process_documents()

if __name__ == "__main__":
    main()
