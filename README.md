# Patent Vectorial Database

This project creates a vector database from patent PDFs, allowing for semantic search and analysis of patent content.
The system converts patent text into numerical vector embeddings using SentenceTransformer, indexes them using FAISS (a vector similarity search library), and allows semantic search through these vectors. The chunks are the documents in your database, and their vector representations enable retrieval based on semantic similarity rather than just keyword matching. The combination of chunking, vectorization, indexing, and semantic search functionality fulfills the core requirements of a vectorial database.

## Features

- Extracts and processes text from patent PDFs
- Handles two-column layout common in patent documents
- Removes line numbers, page numbers, and other irrelevant information
- Chunks texts into manageable segments with overlap
- Creates vector embeddings for semantic search
- Clusters similar patent chunks
- Provides search capabilities to find relevant patent information
- Includes a graphical user interface for easy searching

## Why Vectorial Search for Patents?

Patents represent some of the most valuable technical and intellectual property documents. However, traditional keyword-based search has significant limitations when working with patent documents:

1. **Technical Terminology Variations**: Patents often use different terminology to describe the same concepts, making keyword search unreliable
2. **Intentional Obfuscation**: Patent language can be deliberately complex to broaden claims while hiding from keyword searches
3. **Contextual Understanding**: The meaning of technical terms depends heavily on context, which keyword search doesn't capture
4. **Cross-Domain Applications**: Technologies described in patents may apply to multiple domains not explicitly mentioned

Vectorial search solves these problems by:

- **Understanding Semantics**: Captures the meaning of sentences rather than just matching words
- **Identifying Similar Concepts**: Finds related patents even when terminology differs
- **Preserving Context**: Maintains the contextual relationships between concepts
- **Enabling Concept-Based Search**: Allows searching for ideas rather than specific keywords

This system enables researchers, inventors, and legal professionals to:
- Perform comprehensive prior art searches
- Identify potential infringement risks
- Discover cross-industry applications of technologies
- Map competitive patent landscapes more effectively than traditional search methods

## System Architecture

The system processes patents through several stages:

1. **Text Extraction**: Convert PDF patents to raw text, handling complex layouts
2. **Text Cleaning**: Remove irrelevant content (headers, footers, paragraph numbers)
3. **Chunking**: Split text into manageable segments, respecting sentence boundaries
4. **Vectorization**: Convert text chunks into numerical embeddings using SentenceTransformer
5. **Indexing**: Store vectors in a FAISS index for efficient similarity search
6. **Search**: Query the index to retrieve semantically similar patent chunks
7. **UI**: Present results in a user-friendly interface with links to original PDFs

## Folder Structure

```
/
├── patent_vectorizer.py     # Main processing script
├── process_single_patent.py # Script for processing a single patent
├── search_patents.py        # Command-line search utility
├── ui_search.py             # Graphical user interface for search
│
├── Patents/                 # Directory for storing patent PDFs
│   ├── US*.pdf              # Patent PDF files
│   └── ...
│
├── Patent_DB/               # Output directory for processed data
│   ├── *.txt                # Extracted text from patents
│   ├── *_chunks.txt         # Chunked text segments
│   ├── embeddings.npy       # Vector embeddings matrix
│   ├── chunk_mapping.csv    # Maps chunks to patents
│   ├── patent_index.faiss   # FAISS vector index
│   ├── cluster_assignments.csv  # Cluster assignments
│   ├── cluster_analysis.txt     # Analysis of clusters
│   └── cluster_visualization.png  # t-SNE visualization
│
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For text extraction, you need to install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `apt-get install tesseract-ocr`

## Usage

### 1. Process Patent PDFs and Create Vector Database

#### Process an Entire Directory

Place your patent PDFs in the `Patents` directory and run:

```bash
python patent_vectorizer.py
```

This will:
- Extract text from all PDFs in the `Patents` directory
- Clean and chunk the text
- Generate vector embeddings
- Cluster similar chunks
- Save the results to the `Patent_DB` directory

The process may take some time depending on the number and size of PDFs.

#### Process a Single Patent

To process just one patent PDF file:

```bash
python process_single_patent.py path/to/patent.pdf
```

Options:
- `-o` or `--output`: Specify output directory (default: <filename>_processed)
- `-v` or `--visualize`: Generate embeddings and visualization (takes longer)

Example:
```bash
python process_single_patent.py Patents/US20250051404A1.pdf -v
```

### 2. Search the Vector Database

#### Using the GUI

For a user-friendly interface, run:

```bash
python ui_search.py
```

This opens a graphical interface where you can:
- Enter search queries
- Set the number of results to return
- View formatted results with clickable patent links to open PDFs
- Export results to a file

#### Using the Command Line

You can also search from the command line:

```bash
python search_patents.py "your search query here"
```

Options:
- `-k` or `--top`: Number of results to return (default: 5)
- `-d` or `--db`: Path to the database directory (default: Patent_DB)
- `-o` or `--output`: Save results to a file

Example:
```bash
python search_patents.py "energy storage system" -k 10 -o results.txt
```

## Vectorization Process Details

The vectorization process involves multiple steps to ensure high-quality search results:

1. **PDF Processing**: 
   - Extracts text while preserving document structure using PyMuPDF
   - Handles multi-column layouts common in patent documents
   - Preserves paragraph and section structure

2. **Sentence-Based Chunking**:
   - Divides documents into chunks while respecting sentence boundaries
   - Includes configurable overlap between chunks to maintain context
   - Prevents loss of information at chunk boundaries

3. **Embedding Model**:
   - Uses SentenceTransformer's `all-MiniLM-L6-v2` model (384-dimensional vectors)
   - Captures semantic meaning rather than just keywords
   - Optimized for performance and accuracy balance

4. **Similarity Search**:
   - Uses FAISS (Facebook AI Similarity Search) for efficient vector search
   - Identifies semantically similar patents regardless of terminology
   - Handles searches across thousands of patent chunks in milliseconds

## Advanced Usage

You can also use the classes programmatically:

```python
from patent_vectorizer import PatentVectorizer

# Create vectorizer
vectorizer = PatentVectorizer(pdf_dir='Patents', output_dir='Patent_DB')

# Process patents
vectorizer.process_all_patents()

# Access data
patent_text = vectorizer.patent_texts['US12345678']

# Search
results = vectorizer.search_patents("energy storage")
for result in results:
    print(f"Patent: {result['patent_id']}, Relevance: {1.0/(1.0+result['distance']):.4f}")
    print(result['text'][:200])
``` 

## Future Improvements

- Integration with public patent databases like USPTO and Google Patents
- Enhanced visualization of patent relationships
- Fine-tuning of embedding models specifically for patent language
- Multi-language support for international patents
- Named entity recognition for chemicals, methods, and apparatus 