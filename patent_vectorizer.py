import os
import re
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
import faiss
from tqdm import tqdm
import textwrap

# Download needed NLTK data
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Warning: Could not download NLTK punkt data: {str(e)}")

class PatentVectorizer:
    def __init__(self, pdf_dir='Patents', output_dir='Patent_DB'):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.patent_texts = {}
        self.patent_chunks = {}
        self.patent_embeddings = {}
        self.faiss_index = None
        self.chunk_mapping = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF handling two-column layout."""
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text with layout recognition
            blocks = page.get_text("dict")["blocks"]
            page_text = ""
            
            # Sort blocks by x0 and y0 to handle columns
            # This puts left column first, then right column
            blocks = sorted(blocks, key=lambda b: (b["bbox"][0], b["bbox"][1]))
            
            for block in blocks:
                if block.get("type") == 0:  # Type 0 is text
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        # Skip if line contains only line numbers, page numbers or empty
                        if re.match(r'^\s*\d+\s*$', line_text) or re.match(r'^\s*$', line_text):
                            continue
                        page_text += line_text + " "
                    page_text += "\n"
            
            full_text += page_text + "\n\n"
        
        # Clean up the text
        full_text = self._clean_text(full_text)
        return full_text
    
    def _clean_text(self, text):
        """Clean the extracted text by removing line numbers, headers/footers, etc."""
        # Remove line numbers (typically appearing at start of line)
        text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)
        
        # Remove page numbers
        text = re.sub(r'\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove headers and footers (often contains patent numbers, dates, etc.)
        text = re.sub(r'^.*patent.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*publication.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove paragraph numbers in brackets (like [0046])
        text = re.sub(r'\[\s*\d+\s*\]', '', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def process_all_patents(self):
        """Process all patents in the directory."""
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        
        for pdf_file in tqdm(pdf_files, desc="Processing patents"):
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            print(f"Processing: {pdf_file}")
            try:
                text = self.extract_text_from_pdf(pdf_path)
                patent_id = os.path.splitext(pdf_file)[0]
                self.patent_texts[patent_id] = text
                
                # Save extracted text to file
                text_file = os.path.join(self.output_dir, f"{patent_id}.txt")
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"  Saved text to {text_file}")
            except Exception as e:
                print(f"  Error processing {pdf_file}: {str(e)}")
    
    def chunk_patents(self, chunk_size=1000, overlap=200):
        """Split patent texts into smaller chunks with overlap, respecting sentence boundaries."""
        for patent_id, text in self.patent_texts.items():
            # Remove paragraph numbers in brackets before tokenizing into sentences
            cleaned_text = re.sub(r'\[\s*\d+\s*\]', '', text)
            
            # Extract sentences from the text
            try:
                # Try NLTK first for better sentence boundary detection
                sentences = sent_tokenize(cleaned_text)
                print(f"  NLTK tokenizer found {len(sentences)} sentences for {patent_id}")
            except Exception as e:
                print(f"  NLTK tokenization failed for {patent_id}: {str(e)}")
                # Fallback to custom sentence splitting with regex
                sentences = []
                # Split on periods, question marks, and exclamation points followed by space or newline
                sentence_parts = re.split(r'(?<=[.!?])(?=\s|$)', cleaned_text)
                for part in sentence_parts:
                    part = part.strip()
                    if len(part) > 10:  # Avoid very short fragments
                        sentences.append(part)
                print(f"  Custom tokenizer found {len(sentences)} sentences for {patent_id}")
            
            if not sentences:
                print(f"  Warning: No sentences found for {patent_id}, using text paragraphs instead")
                # Fallback to paragraph splitting
                sentences = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
            
            # Group sentences into chunks while respecting sentence boundaries
            chunks = []
            current_chunk_sentences = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed the chunk size and we already have content
                if current_length + sentence_length > chunk_size and current_chunk_sentences:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk_sentences)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    if overlap > 0:
                        # Calculate how many sentences to keep for overlap
                        overlap_sentences = []
                        overlap_length = 0
                        
                        # Add sentences from end of previous chunk for overlap
                        for prev_sentence in reversed(current_chunk_sentences):
                            if overlap_length + len(prev_sentence) > overlap:
                                # We've reached our overlap target
                                break
                            overlap_sentences.insert(0, prev_sentence)
                            overlap_length += len(prev_sentence) + 1  # +1 for space
                        
                        # Start new chunk with overlap sentences
                        current_chunk_sentences = overlap_sentences
                        current_length = overlap_length
                    else:
                        # No overlap
                        current_chunk_sentences = []
                        current_length = 0
                
                # Handle case where a single sentence is longer than chunk_size
                if sentence_length > chunk_size and not current_chunk_sentences:
                    # Just add this long sentence as its own chunk
                    chunks.append(sentence)
                    current_chunk_sentences = []
                    current_length = 0
                    continue
                
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            
            # Add the last chunk if not empty
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
            
            print(f"  Created {len(chunks)} chunks for {patent_id}")
            self.patent_chunks[patent_id] = chunks
            
            # Save chunks to file
            chunk_file = os.path.join(self.output_dir, f"{patent_id}_chunks.txt")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"CHUNK {i+1}:\n")
                    wrapped_text = textwrap.fill(chunk, width=100)
                    f.write(f"{wrapped_text}\n\n{'='*80}\n\n")
    
    def process_single_patent(self, pdf_path, output_dir=None):
        """Process a single patent file."""
        if output_dir is None:
            # Use the filename without extension as output directory
            filename = os.path.basename(pdf_path)
            patent_id = os.path.splitext(filename)[0]
            output_dir = os.path.join(self.output_dir, f"{patent_id}_processed")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Processing: {pdf_path}")
        try:
            text = self.extract_text_from_pdf(pdf_path)
            patent_id = os.path.basename(os.path.splitext(pdf_path)[0])
            self.patent_texts[patent_id] = text
            
            # Save extracted text to file
            text_file = os.path.join(output_dir, f"{patent_id}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"  Saved text to {text_file}")
            
            # Chunk the patent text
            self.chunk_patents(chunk_size=1000, overlap=200)
            
            # Save chunks to the specified output directory
            if patent_id in self.patent_chunks:
                chunk_file = os.path.join(output_dir, f"{patent_id}_chunks.txt")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(self.patent_chunks[patent_id]):
                        f.write(f"CHUNK {i+1}:\n")
                        wrapped_text = textwrap.fill(chunk, width=100)
                        f.write(f"{wrapped_text}\n\n{'='*80}\n\n")
                
                print(f"  Saved chunks to {chunk_file}")
            
            return output_dir
        except Exception as e:
            print(f"  Error processing {pdf_path}: {str(e)}")
            return None
    
    def generate_embeddings(self):
        """Generate embeddings for each patent chunk."""
        all_chunks = []
        chunk_to_patent = []
        
        # Collect all chunks and track their source
        for patent_id, chunks in self.patent_chunks.items():
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_to_patent.append((patent_id, chunk_idx))
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Generating embeddings"):
            batch = all_chunks[i:i+batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Store mapping for retrieval
        self.chunk_mapping = chunk_to_patent
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)
        
        # Save embeddings and mapping
        np.save(os.path.join(self.output_dir, 'embeddings.npy'), embeddings_array)
        pd.DataFrame(self.chunk_mapping, columns=['patent_id', 'chunk_idx']).to_csv(
            os.path.join(self.output_dir, 'chunk_mapping.csv'), index=False
        )
        
        # Save faiss index
        faiss.write_index(self.faiss_index, os.path.join(self.output_dir, 'patent_index.faiss'))
    
    def cluster_embeddings(self, n_clusters=10):
        """Cluster the embeddings and visualize."""
        if self.faiss_index is None:
            print("No embeddings found. Run generate_embeddings first.")
            return
        
        # Get embeddings from FAISS index
        embeddings = faiss.extract_index_data(self.faiss_index)[0]
        
        # Run K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1], 
            c=cluster_labels, 
            cmap='viridis', 
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Patent Chunks Clustered into {n_clusters} Groups')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_visualization.png'), dpi=300)
        
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'patent_id': [m[0] for m in self.chunk_mapping],
            'chunk_idx': [m[1] for m in self.chunk_mapping],
            'cluster': cluster_labels
        })
        cluster_df.to_csv(os.path.join(self.output_dir, 'cluster_assignments.csv'), index=False)
        
        # Analyze clusters
        self._analyze_clusters(cluster_df, n_clusters)
    
    def _analyze_clusters(self, cluster_df, n_clusters):
        """Analyze the clusters and create a summary."""
        analysis_file = os.path.join(self.output_dir, 'cluster_analysis.txt')
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(f"Patent Chunk Cluster Analysis\n")
            f.write(f"==============================\n\n")
            
            for cluster_id in range(n_clusters):
                cluster_rows = cluster_df[cluster_df['cluster'] == cluster_id]
                f.write(f"Cluster {cluster_id}: {len(cluster_rows)} chunks\n")
                f.write(f"------------------------------------------------\n")
                
                # Count patents in this cluster
                patents_in_cluster = cluster_rows['patent_id'].unique()
                f.write(f"Patents: {len(patents_in_cluster)}\n")
                
                # Sample some chunks from this cluster
                sample_size = min(5, len(cluster_rows))
                sample_rows = cluster_rows.sample(sample_size, random_state=42)
                
                f.write("\nSample chunks from this cluster:\n\n")
                
                for _, row in sample_rows.iterrows():
                    patent_id = row['patent_id']
                    chunk_idx = row['chunk_idx']
                    chunk_text = self.patent_chunks[patent_id][chunk_idx]
                    
                    f.write(f"Patent: {patent_id}\n")
                    f.write(f"Chunk: {chunk_idx+1}\n")
                    # Get first 300 characters as a preview
                    preview = chunk_text[:300] + '...' if len(chunk_text) > 300 else chunk_text
                    preview = textwrap.fill(preview, width=80)
                    f.write(f"{preview}\n\n")
                
                f.write(f"\n{'='*50}\n\n")
    
    def search_patents(self, query, k=5):
        """Search the vectorial database for similar patents."""
        if self.faiss_index is None:
            print("No index found. Run generate_embeddings first.")
            return
        
        # Generate embedding for the query
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search the FAISS index
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunk_mapping):
                patent_id, chunk_idx = self.chunk_mapping[idx]
                chunk_text = self.patent_chunks[patent_id][chunk_idx]
                
                results.append({
                    'patent_id': patent_id,
                    'chunk_idx': chunk_idx,
                    'distance': distances[0][i],
                    'text': chunk_text[:300] + '...' if len(chunk_text) > 300 else chunk_text
                })
        
        return results

def main():
    print("Initializing Patent Vectorizer...")
    vectorizer = PatentVectorizer()
    
    print("\nProcessing all patent PDFs...")
    vectorizer.process_all_patents()
    
    print("\nChunking patent texts...")
    vectorizer.chunk_patents()
    
    print("\nGenerating embeddings...")
    vectorizer.generate_embeddings()
    
    print("\nClustering embeddings...")
    vectorizer.cluster_embeddings(n_clusters=15)
    
    print("\nVectorial database created successfully. You can now search using:")
    print("from patent_vectorizer import PatentVectorizer")
    print("vectorizer = PatentVectorizer()")
    print("vectorizer.search_patents('your query here')")

if __name__ == "__main__":
    main() 