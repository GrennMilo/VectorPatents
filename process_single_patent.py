import os
import sys
import argparse
import traceback
from patent_vectorizer import PatentVectorizer

def process_single_patent(pdf_path, output_dir=None, visualize=False):
    """Process a single patent PDF file."""
    # Ensure the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return False
    
    print(f"Processing file: {pdf_path}")
    
    # Extract file name without extension
    file_name = os.path.basename(pdf_path)
    file_base = os.path.splitext(file_name)[0]
    
    print(f"File base name: {file_base}")
    
    # Determine output directory
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(pdf_path), f"{file_base}_processed")
    
    print(f"Output directory: {output_dir}")
    
    try:
        # Initialize the vectorizer and process the single patent
        print("Initializing PatentVectorizer...")
        vectorizer = PatentVectorizer()
        print("PatentVectorizer initialized successfully")
        
        # Process the patent using the new method
        print(f"Processing patent: {file_name}")
        result_dir = vectorizer.process_single_patent(pdf_path, output_dir)
        
        if not result_dir:
            print("Error: Patent processing failed")
            return False
        
        # Generate embeddings (only if we want visualization)
        if visualize:
            print("Generating embeddings...")
            vectorizer.generate_embeddings()
            
            # Generate visualization with clustering
            print("Generating visualization...")
            try:
                # Get the number of chunks for this patent
                n_chunks = len(vectorizer.patent_chunks[file_base])
                # Use minimum of 5 clusters or the number of chunks
                vectorizer.cluster_embeddings(n_clusters=min(5, max(1, n_chunks)))
            except Exception as viz_error:
                print(f"Warning: Could not generate clusters/visualization: {str(viz_error)}")
                traceback.print_exc()
        
        print(f"\nProcessing complete. Results saved to {result_dir}")
        return True
    
    except Exception as e:
        print(f"Error processing patent: {str(e)}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a single patent PDF file')
    parser.add_argument('pdf_path', help='Path to the patent PDF file')
    parser.add_argument('-o', '--output', help='Output directory (default: <filename>_processed)')
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help='Generate embeddings and visualization (takes longer)')
    
    args = parser.parse_args()
    
    print(f"Arguments: pdf_path={args.pdf_path}, output={args.output}, visualize={args.visualize}")
    
    try:
        success = process_single_patent(args.pdf_path, args.output, args.visualize)
        
        if not success:
            print("Processing failed.")
            sys.exit(1)
        else:
            print("Processing successful.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 