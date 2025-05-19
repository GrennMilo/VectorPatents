import nltk

def download_nltk_data():
    print("Downloading NLTK data...")
    
    # Download the punkt tokenizer
    nltk.download('punkt')
    
    # Download other NLTK data as needed
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    
    print("NLTK data download complete.")

if __name__ == "__main__":
    download_nltk_data() 