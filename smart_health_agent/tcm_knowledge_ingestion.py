#!/usr/bin/env python3
"""
TCM Knowledge Ingestion Script

This script ingests Traditional Chinese Medicine (TCM) knowledge from the markdown file
into the Milvus vector database for Retrieval Augmented Generation (RAG) in the
Smart Health Agent system.

Usage:
    python tcm_knowledge_ingestion.py [--force] [--chunk-size CHUNK_SIZE] [--overlap OVERLAP]

Options:
    --force: Force re-ingestion even if TCM knowledge already exists
    --chunk-size: Size of text chunks (default: 1000)
    --overlap: Overlap between chunks (default: 200)
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import MilvusClient
from document_processor import chunk_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tcm_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TCMKnowledgeIngestion:
    """Handles ingestion of TCM knowledge into Milvus vector database."""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530", collection_name: str = "health_knowledge"):
        """
        Initialize the TCM knowledge ingestion system.
        
        Args:
            milvus_uri: Milvus server URI
            collection_name: Name of the Milvus collection
        """
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # TCM knowledge file path
        self.tcm_knowledge_file = Path(__file__).parent / "tcm_knowledge_base.md"
        
    def check_tcm_knowledge_exists(self) -> bool:
        """
        Check if TCM knowledge already exists in the collection.
        
        Returns:
            True if TCM knowledge exists, False otherwise
        """
        try:
            # Query for documents with TCM source
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="source == 'tcm_knowledge_base.md'",
                output_fields=["source", "content"],
                limit=1
            )
            return len(results) > 0
        except Exception as e:
            logger.warning(f"Could not check for existing TCM knowledge: {e}")
            return False
    
    def load_tcm_knowledge(self) -> str:
        """
        Load TCM knowledge from the markdown file.
        
        Returns:
            Content of the TCM knowledge file
        """
        if not self.tcm_knowledge_file.exists():
            raise FileNotFoundError(f"TCM knowledge file not found: {self.tcm_knowledge_file}")
        
        try:
            with open(self.tcm_knowledge_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded TCM knowledge from {self.tcm_knowledge_file}")
            return content
        except Exception as e:
            raise Exception(f"Error loading TCM knowledge file: {e}")
    
    def create_tcm_documents(self, content: str) -> List[Document]:
        """
        Create Document objects from TCM knowledge content.
        
        Args:
            content: Raw TCM knowledge content
            
        Returns:
            List of Document objects
        """
        # Create a single document for the entire TCM knowledge base
        tcm_document = Document(
            page_content=content,
            metadata={
                "source": "tcm_knowledge_base.md",
                "type": "tcm_knowledge",
                "title": "Traditional Chinese Medicine Knowledge Base",
                "category": "alternative_medicine",
                "language": "en",
                "version": "1.0"
            }
        )
        
        return [tcm_document]
    
    def chunk_tcm_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Chunk TCM documents into smaller pieces for better RAG retrieval.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunked_docs = chunk_documents(documents, chunk_size, chunk_overlap)
            logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
            return chunked_docs
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            # Fallback to manual chunking
            return self._manual_chunk_documents(documents, chunk_size, chunk_overlap)
    
    def _manual_chunk_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """
        Manual chunking fallback method.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Embed documents using the HuggingFace embeddings model.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        try:
            logger.info(f"Embedding {len(documents)} documents...")
            
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Create embedding records
            embedding_records = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                record = {
                    "id": f"tcm_{i}_{hash(doc.page_content) % 1000000}",
                    "content": doc.page_content,
                    "embedding": embedding,
                    **doc.metadata
                }
                embedding_records.append(record)
            
            logger.info(f"Successfully embedded {len(embedding_records)} documents")
            return embedding_records
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def add_to_milvus(self, embedding_records: List[Dict[str, Any]]) -> None:
        """
        Add embedded documents to Milvus collection.
        
        Args:
            embedding_records: List of embedding records
        """
        try:
            logger.info(f"Adding {len(embedding_records)} TCM knowledge records to Milvus...")
            
            # Insert records into Milvus
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=embedding_records
            )
            
            logger.info(f"Successfully added {len(embedding_records)} TCM knowledge records to Milvus")
            
        except Exception as e:
            logger.error(f"Error adding documents to Milvus: {e}")
            raise
    
    def remove_existing_tcm_knowledge(self) -> None:
        """
        Remove existing TCM knowledge from the collection.
        """
        try:
            logger.info("Removing existing TCM knowledge from collection...")
            
            # Delete documents with TCM source
            self.milvus_client.delete(
                collection_name=self.collection_name,
                filter="source == 'tcm_knowledge_base.md'"
            )
            
            logger.info("Successfully removed existing TCM knowledge")
            
        except Exception as e:
            logger.error(f"Error removing existing TCM knowledge: {e}")
            raise
    
    def ingest_tcm_knowledge(self, force: bool = False, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Main method to ingest TCM knowledge into Milvus.
        
        Args:
            force: Force re-ingestion even if TCM knowledge already exists
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        try:
            logger.info("Starting TCM knowledge ingestion process...")
            
            # Check if TCM knowledge already exists
            if not force and self.check_tcm_knowledge_exists():
                logger.info("TCM knowledge already exists in collection. Use --force to re-ingest.")
                return
            
            # Remove existing TCM knowledge if forcing re-ingestion
            if force:
                self.remove_existing_tcm_knowledge()
            
            # Load TCM knowledge
            content = self.load_tcm_knowledge()
            
            # Create documents
            documents = self.create_tcm_documents(content)
            
            # Chunk documents
            chunked_documents = self.chunk_tcm_documents(documents, chunk_size, chunk_overlap)
            
            # Embed documents
            embedding_records = self.embed_documents(chunked_documents)
            
            # Add to Milvus
            self.add_to_milvus(embedding_records)
            
            logger.info("TCM knowledge ingestion completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during TCM knowledge ingestion: {e}")
            raise
    
    def verify_ingestion(self) -> bool:
        """
        Verify that TCM knowledge was successfully ingested.
        
        Returns:
            True if verification successful, False otherwise
        """
        try:
            # Query for TCM documents
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="source == 'tcm_knowledge_base.md'",
                output_fields=["source", "content"],
                limit=5
            )
            
            if len(results) > 0:
                logger.info(f"Verification successful: Found {len(results)} TCM knowledge records")
                return True
            else:
                logger.warning("Verification failed: No TCM knowledge records found")
                return False
                
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return False

def main():
    """Main function to run TCM knowledge ingestion."""
    parser = argparse.ArgumentParser(description="Ingest TCM knowledge into Milvus vector database")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion even if TCM knowledge already exists")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks (default: 1000)")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    parser.add_argument("--milvus-uri", type=str, default="http://localhost:19530", help="Milvus server URI")
    parser.add_argument("--collection", type=str, default="health_knowledge", help="Milvus collection name")
    parser.add_argument("--verify", action="store_true", help="Verify ingestion after completion")
    
    args = parser.parse_args()
    
    try:
        # Initialize ingestion system
        ingestion = TCMKnowledgeIngestion(
            milvus_uri=args.milvus_uri,
            collection_name=args.collection
        )
        
        # Perform ingestion
        ingestion.ingest_tcm_knowledge(
            force=args.force,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap
        )
        
        # Verify if requested
        if args.verify:
            if ingestion.verify_ingestion():
                logger.info("✅ TCM knowledge ingestion verified successfully!")
            else:
                logger.error("❌ TCM knowledge ingestion verification failed!")
                sys.exit(1)
        
        logger.info("🎉 TCM knowledge ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ TCM knowledge ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 