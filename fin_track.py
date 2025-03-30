import traceback
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
import nltk
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page config (must be first Streamlit command)
st.set_page_config(
    page_title="Finance Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Initialize LangChain components
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('transactions_with_types.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Create vector store directly from DataFrame
def create_vector_store(df):
    # Create a combined text field for embedding
    df['combined_text'] = df.apply(
        lambda row: f"Transaction {row['transaction_id']}: {row['amount']} {row['type']} for {row['category_description']} on {row['date']}",
        axis=1
    )
    
    # Create vector store
    vectorstore = FAISS.from_texts(
        df['combined_text'].tolist(),
        embeddings,
        metadatas=df.to_dict('records')
    )
    return vectorstore

vectorstore = create_vector_store(df)

# Create compression retriever with Cohere
compression_retriever = ContextualCompressionRetriever(
    base_compressor=CohereRerank(),
    base_retriever=vectorstore.as_retriever()
)

# Initialize ChatGroq with Llama 70B
model = ChatGroq(
    model_name="Llama3-70B-8192",
    temperature=0.1,
    max_tokens=2048,
    top_p=0.95,
    verbose=True
)

class FinanceTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_spending(self, date: str) -> str:
        """Analyze spending for a specific date"""
        daily_data = self.df[self.df['date'].dt.date == pd.to_datetime(date).date()]
        if daily_data.empty:
            return f"No transactions found for {date}"
        
        total_spent = daily_data['amount'].sum()
        transaction_count = len(daily_data)
        balance = daily_data['balance_left'].iloc[-1]
        
        return f"On {date}, you had {transaction_count} transactions totaling ${total_spent:.2f}. Your balance at the end of the day was ${balance:.2f}"

    def analyze_trends(self, start_date: str, end_date: str) -> str:
        """Analyze spending trends between two dates"""
        mask = (self.df['date'].dt.date >= pd.to_datetime(start_date).date()) & \
               (self.df['date'].dt.date <= pd.to_datetime(end_date).date())
        period_data = self.df[mask]
        
        total_spent = period_data['amount'].sum()
        avg_transaction = period_data['amount'].mean()
        category_trends = period_data.groupby('category_description')['amount'].sum()
        
        return f"Between {start_date} and {end_date}, you spent ${total_spent:.2f} across {len(period_data)} transactions. Average transaction: ${avg_transaction:.2f}"

    def get_balance_info(self) -> str:
        """Get current balance and recent changes"""
        current_balance = self.df['balance_left'].iloc[-1]
        last_week_balance = self.df[self.df['date'] >= pd.Timestamp.now() - pd.Timedelta(days=7)]['balance_left'].iloc[0]
        balance_change = current_balance - last_week_balance
        
        return f"Current balance: ${current_balance:.2f}. Balance change in the last week: ${balance_change:.2f}"

    def search_transactions(self, query: str) -> str:
        """Search for specific transactions"""
        relevant_docs = compression_retriever.get_relevant_documents(query)
        if not relevant_docs:
            return "No relevant transactions found."
        
        results = []
        for doc in relevant_docs[:3]:
            results.append(f"Transaction {doc.metadata['transaction_id']}: ${doc.metadata['amount']} {doc.metadata['type']} for {doc.metadata['category_description']} on {doc.metadata['date']}")
        
        return "\n".join(results)

# Create tools
finance_tools = FinanceTools(df)
tools = [
    Tool(
        name="analyze_spending",
        func=finance_tools.analyze_spending,
        description="Analyze spending for a specific date. Input should be a date in YYYY-MM-DD format."
    ),
    Tool(
        name="analyze_trends",
        func=finance_tools.analyze_trends,
        description="Analyze spending trends between two dates. Input should be two dates in YYYY-MM-DD format separated by a comma."
    ),
    Tool(
        name="get_balance_info",
        func=finance_tools.get_balance_info,
        description="Get current balance and recent changes."
    ),
    Tool(
        name="search_transactions",
        func=finance_tools.search_transactions,
        description="Search for specific transactions based on description or category."
    )
]

# Create prompt template for the agent
AGENT_PROMPT = """You are a helpful financial advisor assistant. You have access to various tools to help analyze financial data and answer questions about transactions, spending patterns, and balances.

Available tools:
{tool_names}

Tool descriptions:
{tools}

When using the tools, follow these guidelines:
1. For date-related queries, use the analyze_spending or analyze_trends tools
2. For balance-related queries, use the get_balance_info tool
3. For specific transaction searches, use the search_transactions tool
4. Always provide clear, concise responses with relevant numbers and dates

Question: {input}

{agent_scratchpad}"""

# Create prompt
prompt = ChatPromptTemplate.from_template(AGENT_PROMPT)

# Create agent with better error handling
agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

# Update agent executor with better error handling
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# Add custom CSS
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 20px;
        padding: 20px 0;
        text-align: center;
        background: linear-gradient(120deg, #1E3D59 0%, #2C5282 100%);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Subheader styling */
    .custom-subheader {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        font-weight: 600;
        margin: 20px 0;
        padding: 10px 0;
        border-bottom: 2px solid #4a5568;
    }
    
    /* Card styling */
    .card {
        background: #2d3748;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    
    /* Analysis text styling */
    .analysis-text {
        font-size: 16px;
        color: #ffffff;
        line-height: 1.6;
        padding: 15px;
        background: #2d3748;
        border-radius: 5px;
        border-left: 4px solid #63b3ed;
    }
    
    /* Plotly chart container */
    .stPlotlyChart {
        background-color: #2d3748;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Date input styling */
    .stDateInput {
        margin: 10px 0;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        border-collapse: collapse;
        width: 100%;
        color: #ffffff;
    }
    
    .dataframe th {
        background-color: #4a5568;
        color: #ffffff;
        font-weight: 600;
        text-align: left;
        padding: 12px;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #4a5568;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #2d3748;
        padding: 15px;
        border-radius: 5px;
        color: #63b3ed;
    }
    
    /* Query input styling */
    .stTextInput {
        margin: 20px 0;
    }
    
    .stTextInput > div > div > input {
        border-radius: 5px;
        border: 2px solid #4a5568;
        padding: 10px;
        font-size: 16px;
        background-color: #2d3748;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #a0aec0;
    }
    
    /* Streamlit default element overrides */
    .streamlit-expanderHeader {
        background-color: #2d3748;
        border-radius: 5px;
    }
    
    .stAlert {
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Dark mode text */
    .st-emotion-cache-uf99v8 {
        color: #ffffff;
    }
    
    /* Dark mode background */
    .st-emotion-cache-18ni7ap {
        background-color: #1a202c;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit interface
st.markdown('<h1 class="main-title">Budget Buddy</h1>', unsafe_allow_html=True)

# Query input with better styling and error handling
st.markdown('<div class="card">', unsafe_allow_html=True)
user_query = st.text_input("💬 Ask me anything about your finances:", 
                          placeholder="Example: How much did I spend last week?")
st.markdown('</div>', unsafe_allow_html=True)

if user_query:
    with st.spinner("Processing your query..."):
        try:
            # First try to get relevant context
            relevant_docs = compression_retriever.get_relevant_documents(user_query)
            context = "\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # Add context to the query
            enhanced_query = f"Based on this context: {context}\n\nUser question: {user_query}"
            
            try:
                response = agent_executor.run(enhanced_query)
                st.markdown(f'<div class="analysis-text">{response}</div>', unsafe_allow_html=True)
            except Exception as agent_error:
                # Fallback to direct tool usage if agent fails
                if "spend" in user_query.lower() and "last week" in user_query.lower():
                    end_date = pd.Timestamp.now().date()
                    start_date = end_date - pd.Timedelta(days=7)
                    response = finance_tools.analyze_trends(str(start_date), str(end_date))
                elif "balance" in user_query.lower():
                    response = finance_tools.get_balance_info()
                else:
                    # Search for relevant transactions as last resort
                    response = finance_tools.search_transactions(user_query)
                
                st.markdown(f'<div class="analysis-text">{response}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error("""Sorry, I couldn't process your query. Please try:
            - Asking about specific dates or date ranges
            - Checking your balance
            - Searching for specific transactions
            - Using simpler, more direct questions""")

# Sidebar for date selection
st.sidebar.markdown('<h2 class="custom-subheader">📅 Date Selection</h2>', unsafe_allow_html=True)
selected_date = st.sidebar.date_input("Select Date", min_value=df['date'].min(), max_value=df['date'].max())

# Date range selection for trend analysis
st.sidebar.markdown('<h2 class="custom-subheader">📊 Trend Analysis</h2>', unsafe_allow_html=True)
default_start_date = pd.to_datetime(selected_date) - pd.Timedelta(days=3)
default_start_date = max(default_start_date, df['date'].min())

start_date = st.sidebar.date_input("Start Date", 
                                 value=default_start_date,
                                 min_value=df['date'].min(), 
                                 max_value=df['date'].max())
end_date = st.sidebar.date_input("End Date", 
                               value=selected_date,
                               min_value=df['date'].min(), 
                               max_value=df['date'].max())

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h2 class="custom-subheader">📊 Daily Financial Overview</h2>', unsafe_allow_html=True)
    analysis = finance_tools.analyze_spending(str(selected_date))
    st.markdown(f'<div class="analysis-text">{analysis}</div>', unsafe_allow_html=True)
    
    # Create visualizations
    daily_data = df[df['date'].dt.date == selected_date]
    if not daily_data.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig1 = px.pie(daily_data, values='amount', names='category_description',
                     title=f'Spending Breakdown for {selected_date}')
        fig1.update_layout(
            title_font=dict(size=20, color='#ffffff', family='Arial'),
            font=dict(family='Arial', color='#ffffff'),
            paper_bgcolor='#2d3748',
            plot_bgcolor='#2d3748',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#ffffff')
            )
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h2 class="custom-subheader">📈 Trend Analysis</h2>', unsafe_allow_html=True)
    if start_date and end_date:
        trends = finance_tools.analyze_trends(str(start_date), str(end_date))
        st.markdown(f'<div class="analysis-text">{trends}</div>', unsafe_allow_html=True)
        
        # Create trend visualization with daily aggregation
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        period_data = df[mask].copy()
        
        # Aggregate data by date and calculate additional metrics
        daily_data = period_data.groupby(period_data['date'].dt.date).agg({
            'amount': ['sum', 'mean', 'count'],
            'category_description': lambda x: ', '.join(x.unique())
        }).reset_index()
        
        daily_data.columns = ['date', 'total_amount', 'avg_amount', 'transaction_count', 'categories']
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        
        # Calculate y-axis range with padding
        y_min = daily_data['total_amount'].min() * 0.9
        y_max = daily_data['total_amount'].max() * 1.1
        
        # Create an enhanced figure with daily spending
        fig2 = go.Figure()
        
        # Add main spending line
        fig2.add_trace(go.Scatter(
            x=daily_data['date'],
            y=daily_data['total_amount'],
            mode='lines+markers',
            name='Daily Spending',
            line=dict(
                width=2.5,
                color='#1f77b4',  # More professional blue
                shape='linear'  # Changed to linear for clearer trend
            ),
            marker=dict(
                size=8,
                symbol='circle',
                color='#1f77b4',
                line=dict(
                    color='white',
                    width=1
                )
            ),
            hovertemplate=(
                "<b>Date</b>: %{x|%Y-%m-%d}<br>" +
                "<b>Total Spent</b>: $%{y:,.2f}<br>" +
                "<b>Transactions</b>: %{customdata[0]}<br>" +
                "<b>Avg. Transaction</b>: $%{customdata[1]:,.2f}<br>" +
                "<b>Categories</b>: %{customdata[2]}<extra></extra>"
            ),
            customdata=list(zip(
                daily_data['transaction_count'],
                daily_data['avg_amount'],
                daily_data['categories']
            ))
        ))
        
        # Update layout with enhanced styling
        fig2.update_layout(
            title={
                'text': 'Daily Spending Trends',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=20,
                    family='Arial',
                    color='#ffffff'
                )
            },
            xaxis=dict(
                title='Date',
                title_font=dict(size=14, color='#ffffff'),
                tickfont=dict(size=12, color='#ffffff'),
                tickformat='%Y-%m-%d',
                gridcolor='#4a5568',
                showgrid=True,
                zeroline=True,
                zerolinecolor='#4a5568',
                zerolinewidth=1,
                showline=True,
                linecolor='#4a5568',
                linewidth=1,
                ticks='outside',
                ticklen=5
            ),
            yaxis=dict(
                title='Amount ($)',
                title_font=dict(size=14, color='#ffffff'),
                tickfont=dict(size=12, color='#ffffff'),
                gridcolor='#4a5568',
                showgrid=True,
                zeroline=True,
                zerolinecolor='#4a5568',
                zerolinewidth=1,
                showline=True,
                linecolor='#4a5568',
                linewidth=1,
                tickprefix='$',
                tickformat=',.2f',
                range=[y_min, y_max],
                ticks='outside',
                ticklen=5
            ),
            hovermode='x unified',
            plot_bgcolor='#2d3748',
            paper_bgcolor='#2d3748',
            showlegend=False,
            height=400,  # Reduced height for better proportions
            margin=dict(l=60, r=30, t=60, b=50),  # Adjusted margins
            shapes=[
                # Add bottom border
                dict(
                    type='line',
                    xref='paper',
                    yref='paper',
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=0,
                    line=dict(color='#4a5568', width=1)
                ),
                # Add left border
                dict(
                    type='line',
                    xref='paper',
                    yref='paper',
                    x0=0,
                    y0=0,
                    x1=0,
                    y1=1,
                    line=dict(color='#4a5568', width=1)
                )
            ]
        )
        
        # Add hover effects
        fig2.update_traces(
            hoverlabel=dict(
                bgcolor='#2d3748',
                font_size=12,
                font_family='Arial',
                font_color='#ffffff',
                bordercolor='#4a5568'
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# Transaction list
st.markdown('<h2 class="custom-subheader">📝 Transaction Details</h2>', unsafe_allow_html=True)
daily_transactions = df[df['date'].dt.date == selected_date]
if not daily_transactions.empty:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(
        daily_transactions[['transaction_id', 'amount', 'type', 'category_description', 'balance_left']],
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No transactions for selected date.")	   