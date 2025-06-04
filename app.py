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
from litellm import completion
from litellm.integrations.vector_stores import OpenAI_VectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores import OpenAI as OpenAIVectorStore
from langchain.embeddings import OpenAIEmbeddings
from agents import Agent, Runner

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
vector_store_id = os.getenv("VECTOR_STORE_ID")

# Function to create the transport policy agent
def create_transport_policy_agent():
    retriever = OpenAIVectorStore(
        embedding=OpenAIEmbeddings(),
        vectorstore_id=vector_store_id,
    ).as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="policy_docs",
        description="Searches transport policy documents for relevant excerpts to help answer user queries."
    )

    return Agent(
        name="TransportPolicyAdvisor",
        instructions="""
You are an expert transport policy advisor. You answer questions for individuals, landowners, and planning agents. Prioritise:
1. National policy over local policy
2. More recent documents over older ones

Always cite the source:
- Use “Policy T5 of London Plan 2021” if the reference is available
- Use paragraph numbers if not, e.g. “NPPF 2024, Paragraph 116”

If unsure, explain what else the user could provide.
""",
        tools=[retriever_tool],
    )

# Function to initialize chat state
def init_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Function to display the conversation history
def display_conversation():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Function to process user input and get agent response
async def process_user_message(user_message):
    agent = create_transport_policy_agent()
    runner = Runner(agent)
    response = await runner.run(user_message)

    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    with st.chat_message("assistant"):
        st.markdown(response.content)
        if hasattr(response, "sources"):
            for doc in response.sources:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page', '')}")
                st.write(doc.page_content)

# Streamlit UI
st.set_page_config(page_title="Transport Policy Advisor", layout="wide")
st.title("Transport Policy Advisor")

init_chat_state()
display_conversation()

if user_input := st.chat_input("Ask a transport policy question..."):
    asyncio.run(process_user_message(user_input))
