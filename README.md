# BudgetBuddy - AI-Powered Financial Assistant

BudgetBuddy is an intelligent financial tracking application that helps you analyze your spending patterns, track expenses, and gain insights into your financial health using natural language processing.

## Features

- **Natural Language Queries**: Ask questions about your finances in plain English
- **Interactive Visualizations**: Dynamic charts and graphs based on your queries
- **Financial Analysis**: Get insights on spending patterns, balance trends, and category breakdowns
- **Responsive UI**: Modern, user-friendly interface

## Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - Groq LLM (Llama3-70b-8192)
  - Cohere Embeddings and Reranking
  - LangChain for orchestration
- **Data Processing**: Pandas, FAISS vector store
- **Visualization**: Plotly

## Setup Instructions

### Prerequisites

- Python 3.8+
- API Keys:
  - Groq API Key
  - Cohere API Key
  - OpenAI API Key (optional)

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/BudgetBuddy.git
   cd BudgetBuddy
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys
   ```
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Generate sample transaction data (optional)
   ```
   python temp.py
   ```

5. Run the application
   ```
   streamlit run fin_track.py
   ```

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Set environment variables in the Streamlit Cloud dashboard:
   - GROQ_API_KEY
   - COHERE_API_KEY
   - OPENAI_API_KEY
5. Deploy!

## File Structure

- `fin_track.py`: Main Streamlit application
- `fin_agent.py`: Agent logic for financial analysis
- `temp.py`: Script to generate synthetic transaction data
- `transactions_with_types.csv`: Sample transaction data
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not included in repo)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 