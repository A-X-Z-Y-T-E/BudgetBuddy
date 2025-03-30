import os
import subprocess
import sys

def main():
    print("Setting up BudgetBuddy...")
    
    # Install requirements
    print("\nInstalling dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("\nCreating .env file...")
        with open('.env', 'w') as f:
            f.write("# API Keys for BudgetBuddy\n")
            f.write("GROQ_API_KEY=\n")
            f.write("COHERE_API_KEY=\n")
            f.write("OPENAI_API_KEY=\n")
        print("Please edit the .env file and add your API keys")
    
    # Download NLTK data
    print("\nDownloading NLTK data...")
    import nltk
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    
    # Check if transactions data exists, if not, generate it
    if not os.path.exists('transactions_with_types.csv'):
        print("\nGenerating sample transaction data...")
        subprocess.check_call([sys.executable, "temp.py"])
    
    print("\nSetup complete! To run the app: streamlit run fin_track.py")

if __name__ == "__main__":
    main() 