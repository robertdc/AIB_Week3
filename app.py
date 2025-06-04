import os
import openai
import asyncio
import pandas as pd
import smtplib
import sqlite3
import json
import streamlit as st
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, handoff, RunContextWrapper

# =========================================================================
# Load environment variables
# =========================================================================
load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
vectordb_api_key = os.getenv("VECTORDB_API_KEY")

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_ENABLED = EMAIL_USER and EMAIL_APP_PASSWORD

DB_FILE = os.getenv("DB_FILE", "leads.db")

EMAIL_ROUTING = {
    "individual": EMAIL_USER,
    "developer": EMAIL_USER,
    "planning_agent": EMAIL_USER
}

LEAD_INFO_CACHE = {}
LEAD_EMAIL_CACHE = {}
EMAIL_DEDUPE_WINDOW = 300

# =========================================================================
# System logging
# =========================================================================
def log_system_message(message):
    if 'system_logs' not in st.session_state:
        st.session_state['system_logs'] = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state['system_logs'].append(f"[{timestamp}] {message}")

# =========================================================================
# MESSAGE PROCESSING
# =========================================================================

async def process_user_message(user_input):
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = ""

    if st.session_state['conversation_history']:
        st.session_state['conversation_history'] += f"\nUser: {user_input}"
    else:
        st.session_state['conversation_history'] = user_input

    log_system_message(f"PROCESSING: New message: {user_input[:50]}...")

    try:
        if 'lead_qualifier' not in st.session_state:
            log_system_message("PROCESSING: Creating lead qualifier agent")
            st.session_state['lead_qualifier'] = create_agent_system()

        log_system_message("PROCESSING: Running through lead qualifier")
        with st.spinner('Processing your message...'):
            result = await Runner.run(
                st.session_state['lead_qualifier'],
                st.session_state['conversation_history']
            )

        response = result.final_output
        log_system_message(f"PROCESSING: Generated response: {response[:50]}...")

        st.session_state['conversation_history'] += f"\nAssistant: {response}"
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state['messages'].append({"role": "assistant", "content": response})

        return response

    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        log_system_message(f"PROCESSING ERROR: {error_msg}")
        return "I apologize, but there was an error processing your message. Please try again."

# =========================================================================
# STREAMLIT MAIN APP
# =========================================================================

def main():
    st.set_page_config(
        page_title="Transport Policy Advisor",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚦 Transport Policy Advisor")
    st.markdown("Chat with our AI advisor to understand which transport policies might apply to your project.")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'system_logs' not in st.session_state:
        st.session_state['system_logs'] = []

    if not init_database():
        st.warning("Failed to initialize database. Check system logs for details.")

    render_sidebar()

    col1, col2 = st.columns([2, 1])

    with col1:
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_input = st.chat_input("Type your message here...")
        if user_input:
            asyncio.run(process_user_message(user_input))
            st.rerun()

    with col2:
        st.subheader("System Logs")
        log_container = st.container(height=500)
        with log_container:
            for log in st.session_state['system_logs']:
                st.text(log)

if __name__ == "__main__":
    main()
