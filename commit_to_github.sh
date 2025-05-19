#!/bin/bash

# Initialize git repository if it doesn't exist
if [ ! -d .git ]; then
    git init
    echo "Git repository initialized"
fi

# Set the remote repository if not already set
if ! git remote | grep -q origin; then
    git remote add origin https://github.com/GrennMilo/VectorPatents.git
    echo "Remote origin added"
fi

# Add all files except those in .gitignore
git add .
echo "Files added to staging area"

# Commit with a descriptive message
git commit -m "Initial commit: Patent Vectorial Database System"
echo "Changes committed"

# Push to GitHub
echo "Attempting to push to GitHub..."
echo "Note: You may need to authenticate with your GitHub credentials"
git push -u origin main

echo "Done! Check if the push was successful."
echo "If you see an error about the 'main' branch, try:"
echo "git branch -M main"
echo "git push -u origin main" 