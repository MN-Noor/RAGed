import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
load_dotenv()
apikey=os.getenv("OPENAI_API")

class MyRetriever:
    def __init__(self, collection_name, persist_directory, model_name, api_key, k=3):
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.retriever_model = ChromadbRM(collection_name=collection_name, persist_directory=persist_directory, embedding_function=default_ef)
        self.llm = dspy.OpenAI(model=model_name, api_key=api_key)
        dspy.settings.configure(lm=self.llm, rm=self.retriever_model)
        self.retrieve = dspy.Retrieve(k=k)

    def search(self, query):
        return self.retrieve(query).passages

def main():
    collection_name = 'Articles'
    persist_directory = './chroma_db'
    model_name = "gpt-3.5-turbo"
    api_key = apikey
    query = "Who is pando"

    retriever = MyRetriever(collection_name, persist_directory, model_name, api_key)
    topK_passages = retriever.search(query)
    print(topK_passages)

if __name__ == "__main__":
    main()