import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Set page configuration first
st.set_page_config(
    page_title="Finance Assistant AI",
    page_icon="💰",
    layout="wide"
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Title and description
st.title("💰 Financial Insights Assistant")
st.markdown("Ask questions about your spending habits or select a date to see your spending breakdown")

# Add custom CSS for better text visibility
st.markdown("""
<style>
    /* Ensure all text is black for better visibility */
    .stTextInput>div>div>input {
        color: #000000 !important;
    }
    
    /* Style for answer container */
    .answer-container {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4361ee;
        margin: 1rem 0;
    }
    
    /* Style for chart containers */
    .chart-container {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the LLM
@st.cache_resource
def get_llm():
    """Initialize the language model"""
    if not groq_api_key:
        st.error("❌ GROQ API key not found. Please add it to your .env file.")
        return None
    return ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="Llama3-8b-8192",
        temperature=0.2,
        streaming=True
    )

# Initialize embeddings for vector store
@st.cache_resource
def get_embeddings():
    """Get embeddings model for vector search"""
    try:
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        st.error(f"Failed to load embeddings model: {e}")
        return None

# Load transaction data
@st.cache_data
def load_transaction_data():
    """Load transaction data from CSV"""
    try:
        df = pd.read_csv("finance_data/transactions_with_types.csv")
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = pd.to_numeric(df['amount'])
        return df
    except Exception as e:
        st.error(f"Error loading transaction data: {str(e)}")
        return None

# Add this function to create documents from transactions
def create_transaction_documents(df):
    """Convert transaction dataframe to documents for RAG"""
    documents = []
    
    # Create a document for each transaction
    for _, row in df.iterrows():
        content = f"""Transaction ID: {row['transaction_id']}
Date: {row['date'].strftime('%Y-%m-%d')}
Amount: ${row['amount']:.2f}
Type: {row['type']}
Category: {row['category_description']}
Balance After Transaction: ${row['balance_left']:.2f}
"""
        metadata = {
            "transaction_id": row['transaction_id'],
            "date": row['date'].strftime('%Y-%m-%d'),
            "amount": float(row['amount']),
            "type": row['type'],
            "category": row['category_description'],
            "balance": float(row['balance_left'])
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    # Add summary documents by category
    for category in df['category_description'].unique():
        category_df = df[df['category_description'] == category]
        total_spent = category_df[category_df['type'] == 'debit']['amount'].sum()
        count = len(category_df[category_df['type'] == 'debit'])
        
        if count > 0:
            content = f"""Category Summary: {category}
Total Spent: ${total_spent:.2f}
Number of Transactions: {count}
Average Transaction: ${total_spent/count:.2f}
First Transaction: {category_df['date'].min().strftime('%Y-%m-%d')}
Latest Transaction: {category_df['date'].max().strftime('%Y-%m-%d')}
"""
            metadata = {
                "document_type": "category_summary",
                "category": category,
                "total_spent": float(total_spent),
                "transaction_count": count
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
    
    # Add monthly summaries
    df['month'] = df['date'].dt.strftime('%Y-%m')
    for month in df['month'].unique():
        month_df = df[df['month'] == month]
        spent = month_df[month_df['type'] == 'debit']['amount'].sum()
        income = month_df[month_df['type'] == 'credit']['amount'].sum()
        
        content = f"""Monthly Summary: {month}
Total Spent: ${spent:.2f}
Total Income: ${income:.2f}
Net Change: ${income - spent:.2f}
Transaction Count: {len(month_df)}
Top Categories: {', '.join(month_df.groupby('category_description')['amount'].sum().sort_values(ascending=False).head(3).index.tolist())}
"""
        metadata = {
            "document_type": "monthly_summary",
            "month": month,
            "total_spent": float(spent),
            "total_income": float(income)
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    return documents

# Add this function to create and cache the vector store
@st.cache_resource
def get_vector_store():
    """Create or load vector store from transaction data"""
    df = load_transaction_data()
    if df is None:
        return None
    
    try:
        # Create documents from transactions
        documents = create_transaction_documents(df)
        
        # Split documents (not really needed for our short documents, but included for completeness)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = get_embeddings()
        if embeddings is None:
            return None
            
        vector_store = FAISS.from_documents(splits, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Define tools for the agent
class BalanceAnalysisTool(BaseTool):
    name: str = "balance_analysis"
    description: str = "Analyze account balance trends over time"
    
    def _run(self, query: str) -> str:
        df = load_transaction_data()
        if df is None:
            return "Error: Transaction data not available"
        
        current_balance = df['balance_left'].iloc[-1]
        starting_balance = df['balance_left'].iloc[0]
        min_balance = df['balance_left'].min()
        max_balance = df['balance_left'].max()
        
        # Calculate balance change
        balance_change = current_balance - starting_balance
        percent_change = (balance_change / starting_balance) * 100
        
        result = f"""Balance Analysis:
        • Current Balance: ${current_balance:.2f}
        • Starting Balance: ${starting_balance:.2f}
        • Minimum Balance: ${min_balance:.2f}
        • Maximum Balance: ${max_balance:.2f}
        • Overall Change: ${balance_change:.2f} ({percent_change:.1f}%)
        
        The balance trend suggests {'growth' if balance_change > 0 else 'decline'} over the analyzed period.
        """
        
        return result

class SpendingCategoryTool(BaseTool):
    name: str = "spending_by_category"
    description: str = "Analyze spending habits by category"
    
    def _run(self, query: str) -> str:
        df = load_transaction_data()
        if df is None:
            return "Error: Transaction data not available"
        
        # Filter for debit transactions only
        debit_df = df[df['type'] == 'debit']
        
        # Aggregate spending by category
        category_spending = debit_df.groupby('category_description')['amount'].agg(['sum', 'count']).reset_index()
        category_spending = category_spending.sort_values('sum', ascending=False)
        
        # Calculate percentages
        total_spending = category_spending['sum'].sum()
        category_spending['percentage'] = (category_spending['sum'] / total_spending) * 100
        
        result = "Spending Analysis by Category:\n\n"
        
        for _, row in category_spending.iterrows():
            result += f"• {row['category_description']}: ${row['sum']:.2f} ({row['percentage']:.1f}% of total) - {int(row['count'])} transactions\n"
        
        # Additional insights
        top_category = category_spending.iloc[0]['category_description']
        top_amount = category_spending.iloc[0]['sum']
        top_percent = category_spending.iloc[0]['percentage']
        
        result += f"\nInsights:\n"
        result += f"• Highest spending category is {top_category} at ${top_amount:.2f} ({top_percent:.1f}% of total spending)\n"
        
        return result

class DateAnalysisTool(BaseTool):
    name: str = "analyze_date"
    description: str = "Analyze spending for a specific date or date range"
    
    def _run(self, date_query: str) -> str:
        df = load_transaction_data()
        if df is None:
            return "Error: Transaction data not available"
        
        # Try to parse the date query
        try:
            # Handle various date formats and queries
            if "-" in date_query:
                target_date = pd.to_datetime(date_query)
                date_df = df[df['date'].dt.date == target_date.date()]
            elif "today" in date_query.lower():
                today = datetime.now().date()
                date_df = df[df['date'].dt.date == today]
            elif "yesterday" in date_query.lower():
                yesterday = (datetime.now() - timedelta(days=1)).date()
                date_df = df[df['date'].dt.date == yesterday]
            else:
                # Try to parse as a plain date
                target_date = pd.to_datetime(date_query)
                date_df = df[df['date'].dt.date == target_date.date()]
        except:
            return f"I couldn't understand the date format: '{date_query}'. Please use YYYY-MM-DD format."
        
        if len(date_df) == 0:
            return f"No transactions found for the specified date: {date_query}"
        
        total_spent = date_df[date_df['type'] == 'debit']['amount'].sum()
        total_income = date_df[date_df['type'] == 'credit']['amount'].sum()
        net_change = total_income - total_spent
        
        # Get transactions by category
        category_data = date_df.groupby(['category_description', 'type'])['amount'].sum().reset_index()
        
        result = f"Date Analysis for {date_df['date'].iloc[0].strftime('%Y-%m-%d')}:\n\n"
        result += f"• Total Spent: ${total_spent:.2f}\n"
        result += f"• Total Income: ${total_income:.2f}\n"
        result += f"• Net Change: ${net_change:.2f}\n\n"
        
        result += "Transactions by Category:\n"
        for _, row in category_data.iterrows():
            transaction_type = "💰 Income" if row['type'] == 'credit' else "💸 Expense"
            result += f"• {row['category_description']} ({transaction_type}): ${row['amount']:.2f}\n"
        
        return result

class AffordabilityTool(BaseTool):
    name: str = "affordability_check"
    description: str = "Check if a purchase of a specific amount is affordable"
    
    def _run(self, amount_query: str) -> str:
        df = load_transaction_data()
        if df is None:
            return "Error: Transaction data not available"
        
        # Extract the amount from the query
        try:
            # Try to find a dollar amount in the query
            import re
            amount_matches = re.findall(r'\$(\d+(?:\.\d+)?)', amount_query)
            if amount_matches:
                amount = float(amount_matches[0])
            else:
                # Try to find just a number
                amount_matches = re.findall(r'(\d+(?:\.\d+)?)', amount_query)
                if amount_matches:
                    amount = float(amount_matches[0])
                else:
                    return "I couldn't determine the purchase amount from your query."
        except:
            return "I couldn't determine the purchase amount from your query."
        
        current_balance = df['balance_left'].iloc[-1]
        avg_daily_spending = df[df['type'] == 'debit']['amount'].mean()
        
        # Calculate days until next expected income
        last_date = df['date'].max()
        today = datetime.now().date()
        
        # Find pattern of income
        income_df = df[df['type'] == 'credit']
        if len(income_df) > 0:
            # Calculate typical days between income
            income_dates = income_df['date'].dt.date.tolist()
            if len(income_dates) > 1:
                avg_income_interval = sum((income_dates[i+1] - income_dates[i]).days 
                                       for i in range(len(income_dates)-1)) / (len(income_dates)-1)
            else:
                avg_income_interval = 30  # Default assumption
                
            last_income_date = income_df['date'].max().date()
            days_since_income = (today - last_income_date).days
            expected_days_to_income = max(0, avg_income_interval - days_since_income)
        else:
            expected_days_to_income = 30  # Default assumption
            
        # Projected balance after purchase
        projected_balance = current_balance - amount
        
        # Expected expenses until next income
        expected_expenses = avg_daily_spending * expected_days_to_income
        
        # Affordability analysis
        is_affordable = projected_balance > expected_expenses
        safety_margin = projected_balance - expected_expenses
        
        result = f"Affordability Analysis for ${amount:.2f} Purchase:\n\n"
        result += f"• Current Balance: ${current_balance:.2f}\n"
        result += f"• Balance After Purchase: ${projected_balance:.2f}\n"
        result += f"• Average Daily Spending: ${avg_daily_spending:.2f}\n"
        result += f"• Estimated Days Until Next Income: {expected_days_to_income:.0f}\n"
        result += f"• Expected Expenses Until Then: ${expected_expenses:.2f}\n"
        result += f"• Safety Margin: ${safety_margin:.2f}\n\n"
        
        if is_affordable:
            if safety_margin > amount:
                affordability = "This purchase appears to be easily affordable."
            else:
                affordability = "This purchase is affordable, but will reduce your financial cushion."
        else:
            affordability = "This purchase may put you at risk of insufficient funds before your next income."
            
        result += f"Conclusion: {affordability}"
        
        return result

class SpendingTrendTool(BaseTool):
    name: str = "spending_trends"
    description: str = "Analyze spending trends over time"
    
    def _run(self, query: str) -> str:
        df = load_transaction_data()
        if df is None:
            return "Error: Transaction data not available"
        
        # Aggregate spending by day
        df['day'] = df['date'].dt.date
        daily_spending = df[df['type'] == 'debit'].groupby('day')['amount'].sum().reset_index()
        
        # Calculate 7-day moving average
        daily_spending['7day_avg'] = daily_spending['amount'].rolling(window=7, min_periods=1).mean()
        
        # Get monthly totals
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_spending = df[df['type'] == 'debit'].groupby('month')['amount'].sum().reset_index()
        
        # Calculate month-over-month change
        if len(monthly_spending) > 1:
            latest_month = monthly_spending.iloc[-1]['amount']
            previous_month = monthly_spending.iloc[-2]['amount']
            month_change = ((latest_month - previous_month) / previous_month) * 100
            mom_txt = f"{month_change:.1f}% {'increase' if month_change > 0 else 'decrease'} from previous month"
        else:
            mom_txt = "Not enough data for month-over-month comparison"
        
        # Get recent trends
        recent_days = 14
        if len(daily_spending) > recent_days:
            recent_avg = daily_spending.iloc[-recent_days:]['amount'].mean()
            previous_avg = daily_spending.iloc[-2*recent_days:-recent_days]['amount'].mean()
            trend_change = ((recent_avg - previous_avg) / previous_avg) * 100
            trend_direction = "up" if trend_change > 0 else "down"
        else:
            trend_change = 0
            trend_direction = "stable"
            
        # Weekly spending pattern
        df['weekday'] = df['date'].dt.day_name()
        weekday_spending = df[df['type'] == 'debit'].groupby('weekday')['amount'].mean().reset_index()
        # Sort by weekday
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_spending['weekday_idx'] = weekday_spending['weekday'].map(lambda x: weekday_order.index(x))
        weekday_spending = weekday_spending.sort_values('weekday_idx')
        
        max_weekday = weekday_spending.loc[weekday_spending['amount'].idxmax()]['weekday']
        min_weekday = weekday_spending.loc[weekday_spending['amount'].idxmin()]['weekday']
        
        result = "Spending Trend Analysis:\n\n"
        
        # Monthly insights
        result += "Monthly Insights:\n"
        for _, row in monthly_spending.tail(3).iterrows():
            result += f"• {row['month']}: ${row['amount']:.2f}\n"
        result += f"• Month-over-Month: {mom_txt}\n\n"
        
        # Recent trend
        result += "Recent Trend:\n"
        result += f"• Last {recent_days} days avg: ${recent_avg:.2f} per day\n"
        result += f"• Trend: Spending trending {trend_direction} ({abs(trend_change):.1f}%)\n\n"
        
        # Weekly pattern
        result += "Weekly Pattern:\n"
        result += f"• Highest spending day: {max_weekday}\n"
        result += f"• Lowest spending day: {min_weekday}\n"
        
        return result

# Add a new RAG retrieval tool
class TransactionRetrievalTool(BaseTool):
    name: str = "transaction_retrieval"
    description: str = "Retrieve information about transactions similar to the query"
    
    def _run(self, query: str) -> str:
        vector_store = get_vector_store()
        if vector_store is None:
            return "Error: Vector store not available"
        
        # Retrieve relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant transaction information found."
        
        result = "Here's what I found about your transactions:\n\n"
        for i, doc in enumerate(docs):
            result += f"--- Document {i+1} ---\n{doc.page_content}\n\n"
        
        return result

# Function to create interactive date selection visualization
def show_date_selection():
    df = load_transaction_data()
    if df is None:
        st.error("Could not load transaction data")
        return
    
    # Group by date and sum amounts by transaction type
    daily_df = df.groupby(['date', 'type'])['amount'].sum().unstack().reset_index()
    
    # Fill missing values with 0
    if 'credit' not in daily_df.columns:
        daily_df['credit'] = 0
    if 'debit' not in daily_df.columns:
        daily_df['debit'] = 0
    
    # Calculate net for each day
    daily_df['net'] = daily_df['credit'] - daily_df['debit']
    
    # Create figure
    fig = go.Figure()
    
    # Add debit bars (negative values)
    fig.add_trace(go.Bar(
        x=daily_df['date'],
        y=-daily_df['debit'],
        name='Expenses',
        marker_color='rgba(219, 64, 82, 0.7)'
    ))
    
    # Add credit bars (positive values)
    fig.add_trace(go.Bar(
        x=daily_df['date'],
        y=daily_df['credit'],
        name='Income',
        marker_color='rgba(64, 145, 108, 0.7)'
    ))
    
    # Add net line
    fig.add_trace(go.Scatter(
        x=daily_df['date'],
        y=daily_df['net'],
        name='Net Flow',
        line=dict(color='rgba(46, 49, 146, 1)', width=1.5),
        marker=dict(size=7)
    ))
    
    # Update layout
    fig.update_layout(
        title='Select a date to see detailed analysis',
        barmode='relative',
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Display the chart (note: we don't store the return value)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a date picker instead of relying on chart clicks
    st.subheader("Select a date to analyze")
    
    # Get min and max dates from the dataframe
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    # Default to most recent date with transactions
    default_date = max_date
    
    # Create date picker
    selected_date = st.date_input(
        "Choose a date",
        value=default_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Analyze the selected date when the button is clicked
    if st.button("Analyze Selected Date"):
        analyze_selected_date(selected_date)

# Function to analyze and display information for the selected date
def analyze_selected_date(date_str):
    df = load_transaction_data()
    date = pd.to_datetime(date_str).date()
    
    # Filter transactions for the selected date
    day_df = df[df['date'].dt.date == date]
    
    if len(day_df) == 0:
        st.info(f"No transactions on {date_str}")
        return
    
    # Create two columns for the detailed view
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader(f"Transactions on {date_str}")
        
        # Display transactions
        for _, tx in day_df.iterrows():
            icon = "➕" if tx['type'] == 'credit' else "➖"
            color = "green" if tx['type'] == 'credit' else "red"
            
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                <h4 style="margin:0">{icon} ${tx['amount']:.2f} - {tx['category_description']}</h4>
                <p style="margin:0; color: gray;">ID: {tx['transaction_id']} | Balance after: ${tx['balance_left']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Daily Summary")
        
        # Calculate totals
        total_spent = day_df[day_df['type'] == 'debit']['amount'].sum()
        total_income = day_df[day_df['type'] == 'credit']['amount'].sum()
        net = total_income - total_spent
        
        # Display metric cards
        st.metric("Total Expenses", f"${total_spent:.2f}")
        st.metric("Total Income", f"${total_income:.2f}")
        st.metric("Net Change", f"${net:.2f}", delta=f"{net:.2f}")
        
        # Category breakdown
        st.subheader("Spending by Category")
        categories = day_df[day_df['type'] == 'debit'].groupby('category_description')['amount'].sum()
        
        if not categories.empty:
            fig = px.pie(
                values=categories.values,
                names=categories.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expenses on this day")

# Create tools for agent
def create_agent_tools():
    return [
        Tool(
            name="spending_by_category",
            func=SpendingCategoryTool().run,
            description="Analyze spending habits by category"
        ),
        Tool(
            name="balance_analysis",
            func=BalanceAnalysisTool().run,
            description="Analyze account balance trends over time"
        ),
        Tool(
            name="analyze_date",
            func=DateAnalysisTool().run,
            description="Analyze spending for a specific date (YYYY-MM-DD format)"
        ),
        Tool(
            name="affordability_check",
            func=AffordabilityTool().run,
            description="Check if you can afford a purchase of a specific amount"
        ),
        Tool(
            name="spending_trends",
            func=SpendingTrendTool().run,
            description="Analyze spending trends over time"
        ),
        Tool(
            name="transaction_retrieval",
            func=TransactionRetrievalTool().run,
            description="Retrieve specific transaction information or find similar transactions"
        )
    ]

# Update the get_finance_agent function
def get_finance_agent():
    llm = get_llm()
    if not llm:
        return None
    
    tools = create_agent_tools()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Add a system message to better use the RAG capabilities
    system_message = """You are a financial assistant that helps users understand their transaction data and financial situation.

For general questions about spending patterns, use the spending_trends tool.
For category-specific questions, use the spending_by_category tool.
For date-specific queries, use the analyze_date tool.
For balance and account health questions, use the balance_analysis tool.
For affordability questions, use the affordability_check tool.
For specific transaction information or to find similar transactions, use the transaction_retrieval tool.

Always use the most specific tool for the question. If you need to retrieve specific transaction details, use the transaction_retrieval tool first before performing analysis."""
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        memory=memory,
        verbose=True,
        max_iterations=3,
        agent_kwargs={"system_message": system_message}
    )
    
    return agent

# Main application
def main():
    # Check if transaction data exists
    data_exists = os.path.exists("finance_data/transactions_with_types.csv")
    if not data_exists:
        st.error("Transaction data not found. Please add your transactions_with_types.csv file to the finance_data directory.")
        return
    
    # Initialize vector store on startup
    with st.spinner("Preparing transaction data..."):
        vector_store = get_vector_store()
    
    # Create tabs for different interaction modes
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Date Explorer"])
    
    with tab1:
        st.subheader("Ask about your finances")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to know about your finances?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get agent response
            with st.chat_message("assistant"):
                response_container = st.empty()
                
                agent = get_finance_agent()
                if agent:
                    st_callback = StreamlitCallbackHandler(response_container)
                    response = agent.run(prompt, callbacks=[st_callback])
                    
                    # Display final response
                    response_container.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    response_container.error("Could not initialize the finance agent. Please check your API keys.")
    
    with tab2:
        st.subheader("Explore Your Spending by Date")
        st.write("Click on any date in the chart below to see detailed transaction information.")
        
        show_date_selection()

if __name__ == "__main__":
    main()
