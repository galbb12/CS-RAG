from openai import OpenAI

class RAG:
    def __init__(self, api_key, base_url, retriever):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.retriever = retriever 

    def check_api_key(self):
        try:
            self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "system", "content": "Testing API key."}]
            )
            print("API key is valid.")
        except Exception as e:
            print(f"API key validation failed: {e}")

    def get_new_api_key(self):
        # Placeholder for API key retrieval logic
        # In a real implementation, this could involve prompting the user or fetching from a secure vault
        new_api_key = input("Enter new API key: ")
        self.client.api_key = new_api_key
        self.check_api_key()

    def extract_keywords(self, text):
        response = self.client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Extract keywords from the following text."},
                {"role": "user", "content": text}
            ]
        )
        keywords = response.choices[0].message.content
        return [kw.strip() for kw in keywords.split(",")]
    
    def retrieve_documents(self, keywords):
        return self.retriever.retrieve(keywords)
    
    def chunk_documents(self, documents):
        # Placeholder for document chunking logic
        return documents
    
    def generate_embeddings(self, chunks):
        # Placeholder for embedding generation logic
        return chunks

    def generate_response(self, user_input):
        retrieved_docs = self.retriever.retrieve(user_input)
        context = "\n".join(retrieved_docs)

        response = self.client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{user_input}\n\nContext:\n{context}"}
            ]
        )
        return response.choices[0].message.content