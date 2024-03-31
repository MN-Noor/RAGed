import os
from dspy import OpenAI, settings
from dspy.retrieve.vectara_rm import VectaraRM
from dotenv import load_dotenv

class VectaraRetriever:
    def __init__(self):
        load_dotenv()
        self.customer_id = os.getenv("VECTARA_CUSTOMER_ID")
        self.corpus_id = os.getenv("VECTARA_CORPUS_ID")
        self.api_key = os.getenv("VECTARA_API_KEY")
        self.retriever_model = self._create_retriever_model()

    def _create_retriever_model(self):
        return VectaraRM(
            vectara_customer_id=self.customer_id,
            vectara_corpus_id=self.corpus_id,
            vectara_api_key=self.api_key,
            k=3  # You can adjust k as needed
        )

    def retrieve_passages(self, query, k=3):
        return self.retriever_model(query, k=k)

def main():
    retriever = VectaraRetriever()

    query = "what are marketing strategies"
    retrieved_passages = retriever.retrieve_passages(query, k=3)
    print(retrieved_passages)

if __name__ == "__main__":
    main()