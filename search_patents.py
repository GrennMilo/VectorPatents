import os
import sys
import argparse
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import textwrap

class PatentSearcher:
    def __init__(self, db_dir='Patent_DB'):
        self.db_dir = db_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index = None
        self.chunk_mapping = None
        self.patent_chunks = {}
        
        # Check if database exists
        if not os.path.exists(db_dir):
            print(f"Error: Database directory {db_dir} does not exist.")
            print("Please run patent_vectorizer.py first to create the database.")
            sys.exit(1)
        
        # Load FAISS index
        index_path = os.path.join(db_dir, 'patent_index.faiss')
        if not os.path.exists(index_path):
            print(f"Error: FAISS index not found at {index_path}")
            print("Please run patent_vectorizer.py first to create the index.")
            sys.exit(1)
        
        # Load chunk mapping
        mapping_path = os.path.join(db_dir, 'chunk_mapping.csv')
        if not os.path.exists(mapping_path):
            print(f"Error: Chunk mapping not found at {mapping_path}")
            sys.exit(1)
        
        # Load FAISS index and chunk mapping
        self.faiss_index = faiss.read_index(index_path)
        self.chunk_mapping = pd.read_csv(mapping_path).values.tolist()
        
        # Load patent chunks
        self._load_patent_chunks()
    
    def _load_patent_chunks(self):
        """Load patent chunks from files."""
        patent_ids = set([m[0] for m in self.chunk_mapping])
        
        for patent_id in patent_ids:
            chunk_file = os.path.join(self.db_dir, f"{patent_id}_chunks.txt")
            
            if not os.path.exists(chunk_file):
                print(f"Warning: Chunk file not found for {patent_id}")
                continue
            
            # Read the chunks file
            with open(chunk_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by separator and process
            chunks = []
            chunk_sections = content.split('='*80)
            
            for section in chunk_sections:
                if not section.strip():
                    continue
                
                # Extract chunk text (remove the CHUNK X: line)
                lines = section.strip().split('\n')
                if lines and lines[0].startswith('CHUNK '):
                    chunk_text = ' '.join(lines[1:])
                    chunks.append(chunk_text)
            
            self.patent_chunks[patent_id] = chunks
    
    def search(self, query, k=5, unique_patents=True):
        """Search for patents matching the query.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            unique_patents (bool): If True, only return the best match from each patent
            
        Returns:
            list: List of results
        """
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Increase k if we want unique patents (we might need to fetch more to get k unique ones)
        search_k = k * 3 if unique_patents else k
        
        # Search the FAISS index
        distances, indices = self.faiss_index.search(query_embedding, search_k)
        
        results = []
        seen_patents = set()  # Track patents we've already included
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_mapping):
                patent_id, chunk_idx = self.chunk_mapping[idx]
                
                # Skip if we've already seen this patent and unique_patents is True
                if unique_patents and patent_id in seen_patents:
                    continue
                
                # Check if patent chunks are loaded
                if patent_id not in self.patent_chunks:
                    print(f"Warning: Chunks for patent {patent_id} not loaded")
                    continue
                
                # Check if chunk index is valid
                if chunk_idx >= len(self.patent_chunks[patent_id]):
                    print(f"Warning: Invalid chunk index {chunk_idx} for patent {patent_id}")
                    continue
                
                chunk_text = self.patent_chunks[patent_id][chunk_idx]
                
                results.append({
                    'patent_id': patent_id,
                    'chunk_idx': chunk_idx,
                    'distance': distances[0][i],
                    'text': chunk_text
                })
                
                # Mark this patent as seen
                seen_patents.add(patent_id)
                
                # If we have enough unique patents, stop
                if unique_patents and len(results) >= k:
                    break
        
        return results
    
    def print_results(self, results, max_preview_length=500):
        """Print search results in a formatted way."""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} relevant patent chunks:\n")
        print("="*80)
        
        for i, result in enumerate(results):
            print(f"Result {i+1} (Patent: {result['patent_id']})")
            print(f"Relevance score: {1.0 / (1.0 + result['distance']):.4f}")
            print("-"*80)
            
            # Get a preview of the text
            preview = result['text'][:max_preview_length] + "..." if len(result['text']) > max_preview_length else result['text']
            preview = textwrap.fill(preview, width=80)
            print(preview)
            
            print("\n" + "="*80 + "\n")
    
    def export_results(self, results, output_file):
        """Export search results to a file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Search Results ({len(results)} matches)\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Result {i+1} (Patent: {result['patent_id']})\n")
                f.write(f"Relevance score: {1.0 / (1.0 + result['distance']):.4f}\n")
                f.write("-"*80 + "\n")
                
                # Full text for export
                wrapped_text = textwrap.fill(result['text'], width=80)
                f.write(wrapped_text + "\n\n")
                
                f.write("="*80 + "\n\n")
        
        print(f"Results exported to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Search the patent vector database')
    parser.add_argument('query', help='The search query')
    parser.add_argument('-k', '--top', type=int, default=5, help='Number of results to return (default: 5)')
    parser.add_argument('-d', '--db', default='Patent_DB', help='Path to the database directory (default: Patent_DB)')
    parser.add_argument('-o', '--output', help='Output file to save results')
    parser.add_argument('--all-chunks', action='store_true', help='Show all matching chunks, even from the same patent')
    
    args = parser.parse_args()
    
    print(f"Searching for: {args.query}")
    searcher = PatentSearcher(db_dir=args.db)
    results = searcher.search(args.query, k=args.top, unique_patents=not args.all_chunks)
    
    searcher.print_results(results)
    
    if args.output:
        searcher.export_results(results, args.output)

if __name__ == "__main__":
    main() 