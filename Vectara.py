import vectara
from dotenv import load_dotenv
import os

class VectaraClient:
    def __init__(self):
        load_dotenv()
        self.customer_id = os.getenv("VECTARA_CUSTOMER_ID")
        self.client_id = os.getenv("VECTARA_CLIENT_ID")
        self.client_secret = os.getenv("VECTARA_CLIENT_SECRET")
        self.api_key = os.getenv("VECTARA_API_KEY")
        self.client = vectara.vectara(self.customer_id, self.api_key, self.client_id, self.client_secret)

    def create_corpus(self, corpus_name):
        return self.client.create_corpus(corpus_name)

    def reset_corpus(self, corpus_id):
        return self.client.reset_corpus(corpus_id)

    def upload_folder(self, corpus_id, folder_path):
        return self.client.upload_folder(corpus_id, folder_path)

    def query(self, corpus_id, query_text, top_k=5):
        return self.client.query(corpus_id, query_text, top_k)

# Example usage:
if __name__ == "__main__":
    vectara_client = VectaraClient()
    corpus_id = vectara_client.create_corpus("torchon")
    # corpus_id = 9  # manual set here
    vectara_client.reset_corpus(corpus_id)
    vectara_client.upload_folder(corpus_id, './data')
    vectara_client.query(corpus_id, 'Vectara allows me to search for anything, right?', top_k=5)
