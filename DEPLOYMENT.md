# Deploying BudgetBuddy to Streamlit Cloud

This guide will help you deploy the BudgetBuddy application to Streamlit Cloud.

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (sign up at https://streamlit.io/cloud)
3. API keys:
   - Groq API Key
   - Cohere API Key
   - OpenAI API Key (optional)

## Deployment Steps

### 1. Prepare Your Repository

1. Make sure your GitHub repository contains all the necessary files:
   - `fin_track.py` (main application)
   - `fin_agent.py` (if you're using it)
   - `temp.py` (to generate sample data)
   - `transactions_with_types.csv` (sample data)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (Streamlit configuration)
   - `.gitignore` (to exclude sensitive files)

2. Ensure your `.gitignore` excludes sensitive files:
   - `.env`
   - `.streamlit/secrets.toml`

### 2. Deploy to Streamlit Cloud

1. Log in to [Streamlit Cloud](https://streamlit.io/cloud)

2. Click on "New app" button

3. Connect to your GitHub repository:
   - Select the repository containing BudgetBuddy
   - Select the branch (usually `main` or `master`)
   - Set the main file path to `fin_track.py`

4. Configure Advanced Settings:
   - Python version: 3.9 or higher
   - Packages: No need to add packages manually, they will be installed from `requirements.txt`

### 3. Set Up Secrets

1. In the Streamlit Cloud dashboard, find your app and click on "Settings" ⚙️

2. Navigate to the "Secrets" section

3. Add your secrets in TOML format:
   ```
   [api_keys]
   GROQ_API_KEY = "your_groq_api_key_here"
   COHERE_API_KEY = "your_cohere_api_key_here"
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

4. Save your secrets

### 4. Deploy and Monitor

1. Click on "Deploy" to start the deployment process

2. Streamlit Cloud will clone your repository, install dependencies, and start the app

3. Monitor the logs for any errors during deployment

4. Once deployed, you'll get a public URL for your application

### 5. Troubleshooting

If you encounter any issues:

1. Check the application logs in the Streamlit Cloud dashboard

2. Common issues:
   - Missing dependencies: Make sure all required packages are in `requirements.txt`
   - Secret configuration: Ensure all required API keys are properly set
   - File paths: Make sure file paths in your code are relative (not absolute)

3. For path-related issues, modify your code to use relative paths:
   ```python
   import os
   
   # Instead of hardcoded paths
   data_path = os.path.join(os.path.dirname(__file__), "transactions_with_types.csv")
   ```

### 6. Updating Your App

1. Make changes to your code and push to GitHub

2. Streamlit Cloud will automatically detect changes and redeploy your app

3. To force a redeployment, you can use the "Reboot" option in the app settings

## Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Secrets Management](https://docs.streamlit.io/library/advanced-features/secrets-management)
- [Streamlit GitHub Repository](https://github.com/streamlit/streamlit) 