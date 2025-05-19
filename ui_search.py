import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import pandas as pd
import subprocess
import platform
from search_patents import PatentSearcher

class PatentSearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Patent Vector Database Search")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        self.searcher = None
        self.db_dir = "Patent_DB"
        self.patent_dir = "Patents"  # Directory containing patent PDFs
        self.search_results = []  # Store search results
        
        # Check if database exists
        if not os.path.exists(self.db_dir):
            messagebox.showwarning(
                "Database Not Found", 
                f"Database directory '{self.db_dir}' not found.\n\n"
                "Please run patent_vectorizer.py first to create the database."
            )
        else:
            try:
                self.searcher = PatentSearcher(db_dir=self.db_dir)
                messagebox.showinfo(
                    "Database Loaded", 
                    f"Successfully loaded database from '{self.db_dir}'."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load database: {str(e)}")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main layout frames
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        options_frame = ttk.Frame(self.root, padding="10")
        options_frame.pack(fill=tk.X)
        
        results_frame = ttk.Frame(self.root, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Search bar and buttons
        ttk.Label(top_frame, text="Search Query:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(top_frame, textvariable=self.query_var, width=50)
        self.query_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.query_entry.bind("<Return>", lambda e: self.search())
        
        self.top_results_var = tk.IntVar(value=5)
        ttk.Label(top_frame, text="Top Results:").grid(row=0, column=2, padx=5)
        top_results_spin = ttk.Spinbox(top_frame, from_=1, to=50, width=5, textvariable=self.top_results_var)
        top_results_spin.grid(row=0, column=3, padx=5)
        
        search_btn = ttk.Button(top_frame, text="Search", command=self.search)
        search_btn.grid(row=0, column=4, padx=5)
        
        export_btn = ttk.Button(top_frame, text="Export Results", command=self.export_results)
        export_btn.grid(row=0, column=5, padx=5)
        
        # Configure grid weights
        top_frame.columnconfigure(1, weight=1)
        
        # Options panel
        self.unique_patents_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, 
            text="Show only one result per patent (excludes duplicate patents)",
            variable=self.unique_patents_var
        ).pack(side=tk.LEFT, padx=5)
        
        # Results area
        results_label_frame = ttk.Frame(results_frame)
        results_label_frame.pack(fill=tk.X, anchor=tk.W)
        
        ttk.Label(results_label_frame, text="Search Results:").pack(side=tk.LEFT)
        ttk.Label(results_label_frame, text=" (Click on blue patent IDs to open PDF files)", foreground="gray").pack(side=tk.LEFT)
        
        # Use Text widget instead of ScrolledText for better tag control
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10)
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure common styling for links 
        self.results_text.tag_configure("link", foreground="blue", underline=1)
        self.results_text.tag_bind("link", "<Enter>", lambda e: self.results_text.config(cursor="hand2"))
        self.results_text.tag_bind("link", "<Leave>", lambda e: self.results_text.config(cursor=""))
        
        self.results_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set initial focus to query entry
        self.query_entry.focus_set()
    
    def search(self):
        """Perform the search in a separate thread to keep UI responsive."""
        query = self.query_var.get().strip()
        if not query:
            messagebox.showinfo("Info", "Please enter a search query.")
            return
        
        if not self.searcher:
            messagebox.showwarning("Warning", "Database not loaded. Cannot perform search.")
            return
        
        # Clear previous results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.search_results = []  # Clear previous results
        
        # Update status
        self.status_var.set(f"Searching for: {query}")
        
        # Start search in new thread
        threading.Thread(target=self._do_search, args=(query,), daemon=True).start()
    
    def _do_search(self, query):
        """Execute search in background thread."""
        try:
            k = self.top_results_var.get()
            unique_patents = self.unique_patents_var.get()
            results = self.searcher.search(query, k=k, unique_patents=unique_patents)
            self.search_results = results  # Store results for reference
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_results(results))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Search error: {str(e)}"))
    
    def _update_results(self, results):
        """Update the results area with search results."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "No results found.")
            self.status_var.set("No results found")
        else:
            result_count = len(results)
            unique_patent_count = len(set(r['patent_id'] for r in results))
            
            if self.unique_patents_var.get():
                self.results_text.insert(tk.END, f"Found {result_count} relevant patent chunks from {unique_patent_count} different patents:\n\n")
            else:
                self.results_text.insert(tk.END, f"Found {result_count} relevant patent chunks from {unique_patent_count} different patents (showing all matching chunks):\n\n")
            
            for i, result in enumerate(results):
                patent_id = result['patent_id']
                
                # Insert result header
                self.results_text.insert(tk.END, f"Result {i+1} (Patent: ")
                
                # Insert clickable patent ID
                link_start = self.results_text.index(tk.INSERT)
                self.results_text.insert(tk.END, patent_id)
                link_end = self.results_text.index(tk.INSERT)
                
                # Create a unique tag for this link
                link_tag = f"link_{i}"
                self.results_text.tag_add(link_tag, link_start, link_end)
                self.results_text.tag_config(link_tag, foreground="blue", underline=1)
                
                # Store patent_id in a way that avoids lambda closure issues
                def make_callback(pid):
                    return lambda e: self.open_patent_pdf(pid)
                
                # Bind click event to this specific tag
                self.results_text.tag_bind(link_tag, "<Button-1>", make_callback(patent_id))
                
                # Also apply the general link styling 
                self.results_text.tag_add("link", link_start, link_end)
                
                # Continue with the rest of the result
                self.results_text.insert(tk.END, f")\n")
                self.results_text.insert(tk.END, f"Relevance score: {1.0 / (1.0 + result['distance']):.4f}\n")
                self.results_text.insert(tk.END, "-"*80 + "\n")
                
                # Get a preview of the text (first 500 chars)
                preview = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                self.results_text.insert(tk.END, preview + "\n\n")
                self.results_text.insert(tk.END, "="*80 + "\n\n")
            
            self.status_var.set(f"Found {result_count} results from {unique_patent_count} patents")
        
        self.results_text.config(state=tk.DISABLED)
        # Scroll to top
        self.results_text.see("1.0")
    
    def open_patent_pdf(self, patent_id):
        """Open the PDF file for the given patent ID."""
        # Look for PDF file with patent_id in the filename
        if not os.path.exists(self.patent_dir):
            messagebox.showwarning("Directory Not Found", f"Patents directory '{self.patent_dir}' not found.")
            return
            
        patent_files = [f for f in os.listdir(self.patent_dir) if f.lower().startswith(patent_id.lower()) and f.lower().endswith('.pdf')]
        
        if not patent_files:
            messagebox.showwarning("PDF Not Found", f"Could not find PDF file for patent {patent_id} in {self.patent_dir} directory.")
            return
        
        pdf_path = os.path.join(self.patent_dir, patent_files[0])
        
        try:
            # Open PDF with default application
            if platform.system() == 'Windows':
                os.startfile(pdf_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', pdf_path], check=True)
            else:  # Linux
                subprocess.run(['xdg-open', pdf_path], check=True)
            
            self.status_var.set(f"Opened PDF: {pdf_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF: {str(e)}")
    
    def _show_error(self, message):
        """Show error message in the UI."""
        messagebox.showerror("Error", message)
        self.status_var.set("Error occurred")
    
    def export_results(self):
        """Export search results to a file."""
        query = self.query_var.get().strip()
        if not query:
            messagebox.showinfo("Info", "Please perform a search first.")
            return
        
        if not self.searcher:
            messagebox.showwarning("Warning", "Database not loaded. Cannot export results.")
            return
        
        # Ask for file location
        output_file = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Search Results"
        )
        
        if not output_file:
            return  # User cancelled
        
        try:
            # Export results
            self.searcher.export_results(self.search_results, output_file)
            
            messagebox.showinfo("Export Successful", f"Results exported to {output_file}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")

def main():
    root = tk.Tk()
    app = PatentSearchUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 