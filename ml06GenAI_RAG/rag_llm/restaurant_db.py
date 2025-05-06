from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import logging
import pandas as pd

# logging.basicConfig(level=logging.INFO)  # Add basic logging


class RestaurantReviewDatabase:
    """
    A class for creating and managing a Chroma database of restaurant reviews
    using Ollama embeddings.
    """

    def __init__(self, csv_filepath, model_name="mxbai-embed-large", db_location="./chrome_langchain_db",
                 collection_name="restaurant_reviews", search_kwargs={"k": 5}):
        """
        Initializes the RestaurantReviewDatabase.

        Args:
            csv_filepath (str): The path to the CSV file containing restaurant reviews.
            model_name (str): The name of the Ollama embedding model to use.
            db_location (str): The directory to store the Chroma database.
            collection_name (str): The name of the Chroma collection.
            search_kwargs (dict): Keyword arguments to pass to the retriever's search function.
        """
        self.csv_filepath = csv_filepath
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.db_location = db_location
        self.collection_name = collection_name
        self.search_kwargs = search_kwargs
        self.vector_store = self._load_or_create_db()
        self.retriever = self.vector_store.as_retriever(search_kwargs=self.search_kwargs)

    def _load_or_create_db(self):
        """
        Loads the Chroma database if it exists, otherwise creates it from the CSV file.

        Returns:
            Chroma: The Chroma vector store.
        """
        add_documents = not os.path.exists(self.db_location)
        vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )
        if add_documents:
            self._add_documents_to_db(vector_store)
        return vector_store

    def _add_documents_to_db(self, vector_store):
        """
        Reads the CSV file, creates Langchain Documents, and adds them to the Chroma database.

        Args:
            vector_store (Chroma): The Chroma vector store to add documents to.
        """
        df = pd.read_csv(self.csv_filepath)
        documents = []
        ids = []
        for i, row in df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
            logging.info("review added to db: " + str(document))
        vector_store.add_documents(documents=documents, ids=ids)
        vector_store.persist()

    def get_retriever(self):
        """
        Returns the retriever associated with the Chroma database.

        Returns:
            BaseRetriever: The retriever object.
        """
        return self.retriever

