"""
MULTI-AGENT TRANSPORT POLICY ADVISOR SYSTEM

ARCHITECTURE:
This system implements a Hierarchical Multi-Agent Pattern with the following components:

1. ROUTER AGENT (Orchestrator)
   - Entry point for all user interactions
   - Classifies user type and routes to appropriate specialist
   - Manages agent handoffs and escalations

2. SPECIALIST AGENTS:
   - Individual Agent: Personal transport needs, accessibility, public transport
   - Developer Agent: Commercial projects, planning applications, regulations
   - Expert Agent: Complex policy analysis, legal interpretations, appeals

3. SHARED MEMORY MECHANISMS:
   - Session Context: User classification, conversation history, agent transitions
   - Knowledge Base: Shared policy documents and regulations via vector store
   - Agent Handoff Notes: Context preservation during agent switches

4. ERROR HANDLING & FALLBACKS:
   - Graceful degradation when agents fail
   - Fallback to general advice when specialists unavailable
   - Timeout handling and retry mechanisms
   - User-friendly error messages

AGENT INTERACTION FLOW:
User Input → Router Agent → Classification → Specialist Agent → Response
                ↓
         Shared Memory Updates
                ↓
    Error Handling & Fallbacks
"""

import os
import asyncio
import streamlit as st
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import aiohttp
import re

# Suppress OpenAI deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openai")

# Load environment variables
load_dotenv()


# Handle both local .env and Streamlit secrets
def get_env_var(key: str) -> str:
    """Get environment variable from .env file or Streamlit secrets"""
    # Try .env first (local development)
    value = os.getenv(key)
    if value:
        return value

    # Try Streamlit secrets (deployment)
    try:
        return st.secrets[key]
    except:
        return None


OPENAI_API_KEY = get_env_var("OPENAI_API_KEY")
VECTOR_STORE_ID = get_env_var("VECTOR_STORE_ID")
BRAVE_API_KEY = get_env_var("BRAVE_API_KEY")

# Set up OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Agent Configuration
AGENT_CONFIG = {
    "router": {
        "name": "Transport Policy Router",
        "role": "Orchestrator",
        "instructions": """You are the main orchestrator for a transport policy advisory system. Your responsibilities:

1. CLASSIFY users into one of these categories:
   - INDIVIDUAL: Personal transport, accessibility, daily travel needs
   - DEVELOPER: Commercial development, planning applications, infrastructure
   - EXPERT: Complex policy issues, legal disputes, appeals, technical regulations

2. ASK 1-2 focused questions to determine user type if unclear

3. When confident, respond with EXACTLY this format:
CLASSIFICATION: [INDIVIDUAL/DEVELOPER/EXPERT]
REASON: [Brief explanation]
HANDOFF_NOTES: [Key context for the specialist agent]

4. Handle ESCALATIONS from specialist agents who need expert help

Be efficient but thorough in classification.""",
        "model": "gpt-4o",
        "tools": [{"type": "file_search"}],
        "use_vector_store": True
    },
    "individual": {
        "name": "Individual Transport Advisor",
        "role": "Personal Transport Specialist",
        "instructions": """You are a friendly transport advisor helping individuals with personal needs. You MUST search the policy documents and cite specific policies in your responses.

EXPERTISE:
- Public transport accessibility and routes
- Walking/cycling infrastructure and safety
- Personal mobility and accessibility needs
- Transport costs, ticketing, and entitlements
- Local transport policies affecting daily life
- Rights as a transport user

APPROACH:
- ALWAYS search the policy documents for relevant information
- REPLACE OpenAI's automatic citations 【x:y†source】 with proper policy citations
- Use conversational, empathetic tone
- Ask about specific location and circumstances
- Provide practical, actionable advice with policy backing

CITATION FORMAT - MANDATORY:
You MUST replace any automatic citations like 【4:0†source】with proper policy references:
- For policies: (Document Name Year, Policy X)
- For paragraphs: (Document Name Year, Para. X)

Examples:
- (London Plan 2021, Policy T5)
- (NPPF 2024, Para. 116)
- (Local Transport Plan 2023, Policy LTP4)
- (Highway Code 2022, Para. 59)

IMPORTANT: 
- When referencing information from documents, identify the actual document name and policy/paragraph number, not the system's automatic citation numbers.
- NEVER leave automatic citations like 【4:4†source】 at the end of sentences
- If you mention document names (like "London Plan 2021"), you should still add the specific policy reference in parentheses
- Replace ALL automatic citations throughout your response, not just some of them

If you cannot find specific information in the policy documents, you should indicate this and state that you are searching broader sources for current information.

ESCALATION: If you encounter complex legal/regulatory issues, respond with:
ESCALATE: EXPERT
REASON: [Why escalation needed]
CONTEXT: [Summary for expert agent]""",
        "model": "gpt-4o",
        "tools": [{"type": "file_search"}],
        "use_vector_store": True
    },
    "developer": {
        "name": "Commercial Developer Advisor",
        "role": "Commercial Development Specialist",
        "instructions": """You are a professional transport consultant for commercial development. You MUST search the policy documents and provide detailed citations for all regulatory advice.

EXPERTISE:
- Transport assessments and planning applications
- Highway adoption (S38/S278 agreements)
- Parking standards and requirements
- Public transport accessibility compliance
- Traffic impact assessments
- Sustainable transport obligations
- Planning policy compliance (NPPF, local plans)
- Developer contributions and funding

APPROACH:
- ALWAYS search the policy documents for current regulations
- REPLACE OpenAI's automatic citations 【x:y†source】 with proper policy citations
- Use professional, technical language
- Focus on regulatory compliance and planning requirements
- Ask about project scale, location, and development type
- Provide commercially-focused solutions with policy backing

CITATION FORMAT - MANDATORY:
You MUST replace any automatic citations like 【4:0†source】with proper policy references:
- For policies: (Document Name Year, Policy X)
- For paragraphs: (Document Name Year, Para. X)

Examples:
- (London Plan 2021, Policy T5)
- (NPPF 2024, Para. 116)
- (Local Transport Plan 2023, Policy LTP4)
- (Planning Practice Guidance 2023, Para. 42)

IMPORTANT: 
- When referencing information from documents, identify the actual document name and policy/paragraph number, not the system's automatic citation numbers.
- NEVER leave automatic citations like 【4:4†source】 at the end of sentences
- If you mention document names (like "London Plan 2021"), you should still add the specific policy reference in parentheses
- Replace ALL automatic citations throughout your response, not just some of them

If you cannot find specific information in the policy documents, you should indicate this and state that you are searching broader sources for current information.

ESCALATION: For complex legal disputes or technical regulations:
ESCALATE: EXPERT
REASON: [Why escalation needed]
CONTEXT: [Summary for expert agent]""",
        "model": "gpt-4o",
        "tools": [{"type": "file_search"}],
        "use_vector_store": True
    },
    "expert": {
        "name": "Transport Policy Expert",
        "role": "Advanced Policy & Legal Specialist",
        "instructions": """You are a senior transport policy expert handling complex issues. You MUST thoroughly search the policy documents and provide comprehensive citations with detailed analysis.

EXPERTISE:
- Advanced regulatory interpretation
- Legal disputes and appeals processes
- Technical policy analysis
- Cross-jurisdictional transport law
- Planning inquiries and tribunals
- Complex infrastructure assessments
- Policy development and reform

APPROACH:
- ALWAYS conduct thorough searches of the policy documents
- REPLACE OpenAI's automatic citations 【x:y†source】 with proper policy citations
- Provide authoritative, detailed analysis
- Explain complex procedures step-by-step with policy backing
- Offer strategic advice for difficult situations
- Use precise legal and technical terminology
- Cross-reference multiple policy documents when applicable

CITATION FORMAT - MANDATORY:
You MUST replace any automatic citations like 【4:0†source】with proper policy references:
- For policies: (Document Name Year, Policy X)
- For paragraphs: (Document Name Year, Para. X)

Examples:
- (London Plan 2021, Policy T5)
- (NPPF 2024, Para. 116)
- (Local Transport Plan 2023, Policy LTP4)
- (Planning Practice Guidance 2023, Para. 42)
- (Highways Act 1980, Section 38)
- (Town and Country Planning Act 1990, Section 106)

IMPORTANT: 
- When referencing information from documents, identify the actual document name and policy/paragraph number, not the system's automatic citation numbers.
- NEVER leave automatic citations like 【4:4†source】 at the end of sentences
- If you mention document names (like "London Plan 2021"), you should still add the specific policy reference in parentheses
- Replace ALL automatic citations throughout your response, not just some of them

If you cannot find specific information in the policy documents, you should indicate this and state that you are searching broader sources for current information.

You handle the most complex cases with comprehensive policy research and detailed documentation.""",
        "model": "gpt-4o",
        "tools": [{"type": "file_search"}],
        "use_vector_store": True
    }
}


# Citation cleaning functions
def clean_openai_citations(response: str) -> str:
    """Aggressively remove and replace OpenAI's automatic citations"""
    import re

    # Pattern to match all OpenAI citations like 【4:3†source】 or 【4:3†Sustainable_Transport_SPD】
    citation_pattern = r'【\d+:\d+†[^】]*】'

    # Find all citations to understand the context
    citations = re.findall(citation_pattern, response)

    # Mapping of common document references to proper citations
    document_mappings = {
        'Sustainable_Transport_SPD': 'Sustainable Transport SPD 2013',
        'Core_Strategy': 'Core Strategy 2012',
        'Mayor_Transport_Strategy': "Mayor's Transport Strategy 2018",
        'London_Plan': 'London Plan 2021',
        'NPPF': 'NPPF 2024',
        'Planning_Practice_Guidance': 'Planning Practice Guidance 2023'
    }

    # Create a simple counter for policy references
    policy_counter = 1

    # Remove all OpenAI citations and replace with clean text
    cleaned_response = re.sub(citation_pattern, '', response)

    # If we removed citations, add a note about policy sources
    if citations:
        # Extract unique document types from the citations
        doc_types = set()
        for citation in citations:
            for doc_key in document_mappings.keys():
                if doc_key in citation:
                    doc_types.add(document_mappings[doc_key])

        if doc_types:
            # Add a clean reference note at the end
            sources_note = "\n\n**Policy Sources Referenced:**\n"
            for doc in sorted(doc_types):
                sources_note += f"- {doc}\n"

            cleaned_response = cleaned_response.rstrip() + sources_note

    return cleaned_response


def advanced_citation_replacement(response: str) -> str:
    """Advanced citation replacement with context awareness"""
    import re

    # First, try the aggressive cleaning
    cleaned = clean_openai_citations(response)

    # Additional pattern matching for any remaining issues
    patterns_to_clean = [
        r'【[^】]*】',  # Any remaining 【】 brackets
        r'\[\d+:\d+†[^\]]*\]',  # Square bracket versions
        r'\(\d+:\d+†[^\)]*\)',  # Round bracket versions
    ]

    for pattern in patterns_to_clean:
        cleaned = re.sub(pattern, '', cleaned)

    # Clean up any double spaces or formatting issues
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Multiple newlines to double

    return cleaned.strip()


# Web Search Functions
async def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web using Brave Search API as fallback when vector store lacks info"""
    if not BRAVE_API_KEY:
        return []

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY
            }

            params = {
                "q": query,
                "count": num_results,
                "offset": 0,
                "mkt": "en-GB",  # UK market for transport policies
                "safesearch": "strict"
            }

            async with session.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers=headers,
                    params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("web", {}).get("results", [])
                else:
                    shared_memory.log_error("web_search", f"Brave API error: {response.status}")
                    return []

    except Exception as e:
        shared_memory.log_error("web_search", f"Web search failed: {str(e)}")
        return []


def check_response_completeness(response: str) -> bool:
    """Check if the assistant response indicates missing information"""
    incomplete_indicators = [
        "cannot find",
        "not available in",
        "not found in the documents",
        "unable to locate",
        "no specific information",
        "documents do not contain",
        "not covered in",
        "limited information",
        "insufficient detail"
    ]

    response_lower = response.lower()
    return any(indicator in response_lower for indicator in incomplete_indicators)


async def enhance_response_with_web_search(original_response: str, user_query: str, agent_type: str) -> str:
    """Enhance assistant response with web search if information is incomplete"""

    if not check_response_completeness(original_response):
        return original_response

    # Construct search query based on agent type and user query
    search_queries = {
        "individual": f"UK public transport policy {user_query} rights accessibility",
        "developer": f"UK planning policy transport {user_query} development requirements",
        "expert": f"UK transport law policy {user_query} regulations planning"
    }

    search_query = search_queries.get(agent_type, f"UK transport policy {user_query}")

    if st.session_state.show_status_updates:
        st.info(f"🔍 Searching web for additional information: {search_query}")

    # Perform web search
    search_results = await search_web(search_query)

    if not search_results:
        return original_response + "\n\n⚠️ Could not find additional information through web search."

    # Format search results
    web_info = "\n\n🌐 **Additional Information from Web Search:**\n\n"

    for i, result in enumerate(search_results[:3], 1):
        title = result.get("title", "No title")
        snippet = result.get("description", "No description")
        url = result.get("url", "")

        web_info += f"**{i}. {title}**\n"
        web_info += f"{snippet}\n"
        web_info += f"Source: {url}\n\n"

    web_info += "*Note: Please verify this web information against official policy documents. Web sources may not reflect the most current policy positions.*"

    return original_response + web_info


# Shared Memory Management
class SharedMemory:
    def __init__(self):
        if "shared_memory" not in st.session_state:
            st.session_state.shared_memory = {
                "user_profile": {},
                "conversation_context": [],
                "agent_handoffs": [],
                "escalation_history": [],
                "error_log": []
            }

    def update_user_profile(self, key: str, value: str):
        st.session_state.shared_memory["user_profile"][key] = value

    def add_context(self, agent: str, message: str, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        st.session_state.shared_memory["conversation_context"].append({
            "agent": agent,
            "message": message,
            "timestamp": timestamp
        })

    def log_handoff(self, from_agent: str, to_agent: str, reason: str, context: str):
        st.session_state.shared_memory["agent_handoffs"].append({
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })

    def log_error(self, error_type: str, message: str, agent: str = None):
        st.session_state.shared_memory["error_log"].append({
            "type": error_type,
            "message": message,
            "agent": agent,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_summary(self) -> str:
        profile = st.session_state.shared_memory["user_profile"]
        recent_context = st.session_state.shared_memory["conversation_context"][-3:]

        summary = f"User Profile: {profile}\n"
        summary += "Recent Context:\n"
        for ctx in recent_context:
            summary += f"- {ctx['agent']}: {ctx['message'][:100]}...\n"

        return summary


# Initialize shared memory
shared_memory = SharedMemory()


# Session State Management
def initialize_session_state():
    defaults = {
        "messages": [],
        "thread_id": None,
        "current_agent": "router",
        "user_type": None,
        "agent_threads": {},
        "escalation_count": 0,
        "show_status_updates": False  # Default to OFF for cleaner UI
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

# Streamlit UI Setup
st.set_page_config(page_title="Transport Policy Advisor", layout="wide")
st.title("🚦 Transport Policy Advisor")
st.markdown("Advanced AI system with specialized agents for different transport policy needs")

# Agent Status Display
agent_info = {
    "router": {"icon": "🔄", "name": "Router Agent", "role": "Classifying and routing queries"},
    "individual": {"icon": "👤", "name": "Individual Advisor", "role": "Personal transport guidance"},
    "developer": {"icon": "🏢", "name": "Developer Advisor", "role": "Commercial development support"},
    "expert": {"icon": "⚖️", "name": "Expert Advisor", "role": "Complex policy & legal analysis"}
}

# Get current agent info (always available for spinner text)
current_info = agent_info.get(st.session_state.current_agent, agent_info["router"])

# Only show agent status if status updates are enabled
if st.session_state.show_status_updates:
    st.info(f"{current_info['icon']} **{current_info['name']}** - {current_info['role']}")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Agent Management Functions
async def create_or_get_assistant(agent_type: str) -> str:
    """Create or retrieve assistant for given agent type with vector store integration"""
    try:
        config = AGENT_CONFIG[agent_type]

        # Try environment variable first
        env_var = f"{agent_type.upper()}_ASSISTANT_ID"
        existing_id = get_env_var(env_var)

        # Debug: Show what we're looking for
        if existing_id:
            if st.session_state.show_status_updates:
                st.info(f"🔍 Found {env_var} in environment: {existing_id}")
            try:
                assistant = await client.beta.assistants.retrieve(existing_id)
                if st.session_state.show_status_updates:
                    st.success(f"✅ Successfully retrieved existing {agent_type} assistant")

                # Check if vector store is attached (if required)
                if config.get("use_vector_store") and VECTOR_STORE_ID:
                    needs_vector_store_update = True
                    if assistant.tool_resources and assistant.tool_resources.file_search:
                        vector_store_ids = assistant.tool_resources.file_search.vector_store_ids or []
                        if VECTOR_STORE_ID in vector_store_ids:
                            needs_vector_store_update = False

                    if needs_vector_store_update:
                        # Update assistant to include vector store
                        await client.beta.assistants.update(
                            assistant_id=assistant.id,
                            tool_resources={
                                "file_search": {
                                    "vector_store_ids": [VECTOR_STORE_ID]
                                }
                            }
                        )
                        if st.session_state.show_status_updates:
                            st.info(f"🔗 Attached vector store to existing {agent_type} assistant")
                return assistant.id
            except Exception as e:
                st.error(f"❌ Failed to retrieve {agent_type} assistant ({existing_id}): {str(e)}")
                st.warning(f"This usually means the assistant was deleted or belongs to a different organization.")
                shared_memory.log_error("assistant_retrieval", f"Failed to retrieve {agent_type}: {str(e)}")
        else:
            if st.session_state.show_status_updates:
                st.info(f"🔍 No {env_var} found in environment variables")

        # Create new assistant
        if st.session_state.show_status_updates:
            st.info(f"🛠️ Creating new {agent_type} assistant...")
        assistant_params = {
            "name": config["name"],
            "instructions": config["instructions"],
            "model": config["model"]
        }

        if "tools" in config:
            assistant_params["tools"] = config["tools"]

        # Add vector store if specified and available
        if config.get("use_vector_store") and VECTOR_STORE_ID:
            assistant_params["tool_resources"] = {
                "file_search": {
                    "vector_store_ids": [VECTOR_STORE_ID]
                }
            }
            if st.session_state.show_status_updates:
                st.info(f"🔗 Creating {agent_type} assistant with vector store integration")
        elif config.get("use_vector_store") and not VECTOR_STORE_ID:
            st.warning(f"⚠️ Vector store ID not found for {agent_type} agent. Add VECTOR_STORE_ID to your .env file.")

        assistant = await client.beta.assistants.create(**assistant_params)

        st.success(f"✅ Created {config['name']}: {assistant.id}")
        st.code(f"{env_var}={assistant.id}", language="bash")
        st.info(f"💡 Add the above line to your .env file to reuse this assistant")

        return assistant.id

    except Exception as e:
        shared_memory.log_error("assistant_creation", f"Failed to create {agent_type}: {str(e)}")
        raise Exception(f"Could not create/retrieve {agent_type} assistant: {str(e)}")


def parse_agent_response(response: str) -> Dict:
    """Parse structured responses from agents"""
    result = {
        "classification": None,
        "reason": None,
        "handoff_notes": None,
        "escalate": None,
        "escalate_reason": None,
        "escalate_context": None,
        "message": response
    }

    lines = response.strip().split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('CLASSIFICATION:'):
            result["classification"] = line.split(':', 1)[1].strip()
        elif line.startswith('REASON:'):
            result["reason"] = line.split(':', 1)[1].strip()
        elif line.startswith('HANDOFF_NOTES:'):
            result["handoff_notes"] = line.split(':', 1)[1].strip()
        elif line.startswith('ESCALATE:'):
            result["escalate"] = line.split(':', 1)[1].strip()
        elif line.startswith('ESCALATE_REASON:') or line.startswith('REASON:'):
            result["escalate_reason"] = line.split(':', 1)[1].strip()
        elif line.startswith('CONTEXT:'):
            result["escalate_context"] = line.split(':', 1)[1].strip()

    return result


async def call_agent(user_input: str, agent_type: str, thread_id: Optional[str] = None) -> Tuple[str, str]:
    """Call specific agent with comprehensive error handling and web search fallback"""
    try:
        # Get assistant ID
        assistant_id = await create_or_get_assistant(agent_type)

        # Create or reuse thread
        if thread_id is None:
            thread = await client.beta.threads.create()
            thread_id = thread.id

        # Add conversation context for non-router agents
        if agent_type != "router":
            context = shared_memory.get_context_summary()
            enhanced_input = f"""Context: {context}

CRITICAL CITATION REQUIREMENT - READ CAREFULLY:

You MUST completely replace ALL automatic citations. Do NOT keep any part of the 【】 format.

WRONG EXAMPLES (DO NOT USE):
❌ 【4:9†Sustainable_Transport_SPD】
❌ 【4:5†Core_Strategy】  
❌ 【4:4†source】
❌ Any citation with 【 and 】 brackets

CORRECT EXAMPLES (USE THIS FORMAT):
✅ (Sustainable Transport SPD 2013, Para. 2.14)
✅ (London Plan 2021, Policy T6.1)
✅ (NPPF 2024, Para. 112)
✅ (Core Strategy 2012, Policy DM8)

INSTRUCTIONS:
1. Write your response normally
2. BEFORE finishing, scan for ANY text containing 【 or 】
3. Replace every instance with proper citation format including year
4. Double-check no 【】 brackets remain

User Query: {user_input}"""
        else:
            enhanced_input = user_input

        # Add user message to thread
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=enhanced_input
        )

        # Create and run the assistant
        run = await client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )

        # Poll with timeout and retry logic
        max_wait_time = 60
        start_time = time.time()
        retry_count = 0
        max_retries = 3

        while run.status not in ["completed", "failed", "cancelled", "expired"]:
            if time.time() - start_time > max_wait_time:
                if retry_count < max_retries:
                    retry_count += 1
                    shared_memory.log_error("timeout_retry", f"Retrying {agent_type} call (attempt {retry_count})")
                    await asyncio.sleep(2)
                    start_time = time.time()  # Reset timer
                else:
                    raise Exception(f"Request timed out after {max_retries} retries")

            await asyncio.sleep(1)
            run = await client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

        if run.status == "completed":
            messages = await client.beta.threads.messages.list(thread_id=thread_id)
            assistant_messages = [m for m in messages.data if m.role == "assistant"]

            if assistant_messages:
                latest_message = assistant_messages[0]
                response_parts = []

                for content in latest_message.content:
                    if hasattr(content, 'text'):
                        response_parts.append(content.text.value)
                    elif hasattr(content, 'type') and content.type == 'text':
                        response_parts.append(content.text.value)
                    else:
                        response_parts.append(str(content))

                response = '\n'.join(response_parts) if response_parts else "No response received."

                # AGGRESSIVE CITATION CLEANING - Always clean OpenAI citations
                response = advanced_citation_replacement(response)

                # Check if we need web search fallback (not for router)
                if agent_type != "router" and BRAVE_API_KEY:
                    response = await enhance_response_with_web_search(response, user_input, agent_type)

                # Log successful interaction
                shared_memory.add_context(agent_type, response[:200])

                return response, thread_id
            else:
                raise Exception("No assistant messages found")
        else:
            raise Exception(f"Assistant run failed with status: {run.status}")

    except Exception as e:
        error_msg = f"Error calling {agent_type}: {str(e)}"
        shared_memory.log_error("agent_call", error_msg, agent_type)

        # Fallback strategy
        fallback_response = generate_fallback_response(agent_type, str(e))
        return fallback_response, thread_id


def generate_fallback_response(agent_type: str, error: str) -> str:
    """Generate appropriate fallback responses when agents fail"""
    fallbacks = {
        "router": "I'm having trouble processing your request right now. Could you please tell me if you're asking about personal transport needs or a commercial development project?",
        "individual": "I'm experiencing some technical difficulties. For immediate help with personal transport questions, you might want to check your local transport authority's website or contact them directly.",
        "developer": "There seems to be a technical issue with my systems. For urgent development-related transport questions, I recommend consulting your local planning authority or a transport consultant.",
        "expert": "I'm currently unable to access my advanced analysis capabilities. For complex policy or legal matters, please consider consulting a qualified transport planning professional or legal advisor."
    }

    base_response = fallbacks.get(agent_type, "I'm experiencing technical difficulties. Please try again in a moment.")
    return f"⚠️ {base_response}\n\n*Technical details: {error}*"


def run_agent_sync(user_input: str, agent_type: str, thread_id: Optional[str] = None) -> Tuple[str, str]:
    """Synchronous wrapper for async agent calls"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, call_agent(user_input, agent_type, thread_id))
                return future.result()
        else:
            return asyncio.run(call_agent(user_input, agent_type, thread_id))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(call_agent(user_input, agent_type, thread_id))
        finally:
            loop.close()


def parse_agent_response(response: str) -> Dict:
    """Parse structured responses from agents with improved robustness"""
    result = {
        "classification": None,
        "reason": None,
        "handoff_notes": None,
        "escalate": None,
        "escalate_reason": None,
        "escalate_context": None,
        "message": response
    }

    # Split by lines and also try to find patterns anywhere in the text
    lines = response.strip().split('\n')
    full_text = response.upper()

    # Look for classification patterns - more flexible matching
    import re

    # Try to find CLASSIFICATION pattern anywhere in the text
    classification_patterns = [
        r'CLASSIFICATION:\s*([A-Z]+)',
        r'CLASSIFICATION\s*=\s*([A-Z]+)',
        r'CLASSIFIED\s+AS\s*:\s*([A-Z]+)',
    ]

    for pattern in classification_patterns:
        match = re.search(pattern, full_text)
        if match:
            result["classification"] = match.group(1).strip()
            break

    # If no pattern found, try line-by-line parsing (original method)
    if not result["classification"]:
        for line in lines:
            line = line.strip()
            if line.startswith('CLASSIFICATION:'):
                result["classification"] = line.split(':', 1)[1].strip()
                break

    # Parse other fields
    for line in lines:
        line = line.strip()
        if line.startswith('REASON:') and not result["reason"]:
            result["reason"] = line.split(':', 1)[1].strip()
        elif line.startswith('HANDOFF_NOTES:'):
            result["handoff_notes"] = line.split(':', 1)[1].strip()
        elif line.startswith('ESCALATE:'):
            result["escalate"] = line.split(':', 1)[1].strip()
        elif line.startswith('ESCALATE_REASON:'):
            result["escalate_reason"] = line.split(':', 1)[1].strip()
        elif line.startswith('CONTEXT:'):
            result["escalate_context"] = line.split(':', 1)[1].strip()

    return result


# Main Chat Interface
if user_input := st.chat_input("Ask your transport policy question..."):
    # Validate setup
    if not OPENAI_API_KEY:
        st.error("🔑 OpenAI API key not found. Please check your .env file.")
        st.stop()

    if not VECTOR_STORE_ID:
        st.warning(
            "📚 Vector store ID not found. Add VECTOR_STORE_ID to your .env file to enable policy document access.")
        st.info("Your vector store ID should look like: vs_abc123...")

    if not BRAVE_API_KEY:
        st.info("🔍 Web search not enabled. Add BRAVE_API_KEY to your .env file for web search fallback.")

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(f"🤔 {current_info['name']} is thinking..."):

            # Get appropriate thread ID
            current_thread = st.session_state.agent_threads.get(st.session_state.current_agent)

            # Call current agent
            response, thread_id = run_agent_sync(user_input, st.session_state.current_agent, current_thread)

            # Update thread tracking
            st.session_state.agent_threads[st.session_state.current_agent] = thread_id

            # Parse response for special instructions
            parsed = parse_agent_response(response)

            # Handle router classification
            if st.session_state.current_agent == "router" and parsed["classification"]:
                classification = parsed["classification"].lower()
                if classification in ["individual", "developer", "expert"]:
                    # Log handoff
                    shared_memory.log_handoff(
                        "router",
                        classification,
                        parsed["reason"] or "User classification",
                        parsed["handoff_notes"] or "Initial routing"
                    )

                    # Update user profile
                    shared_memory.update_user_profile("type", classification)

                    # Switch agents
                    st.session_state.current_agent = classification
                    st.session_state.user_type = classification

                    # Show classification ONLY if status updates are enabled
                    if st.session_state.show_status_updates:
                        st.success(f"✅ Classified as: {classification.title()}")
                        if parsed["reason"]:
                            st.info(f"📝 Reason: {parsed['reason']}")

                        # Show router classification in message history only if status updates enabled
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    # Get response from specialist (THIS SHOULD ALWAYS HAPPEN)
                    specialist_thread = st.session_state.agent_threads.get(classification)
                    specialist_response, new_thread = run_agent_sync(
                        user_input, classification, specialist_thread
                    )
                    st.session_state.agent_threads[classification] = new_thread

                    # Always show the specialist response (regardless of status updates setting)
                    st.markdown(specialist_response)
                    st.session_state.messages.append({"role": "assistant", "content": specialist_response})

                else:
                    # Continue with router - show the router response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # Handle escalation requests
            elif parsed["escalate"]:
                escalate_to = parsed["escalate"].lower()
                if escalate_to == "expert":
                    if st.session_state.show_status_updates:
                        st.warning("🔄 Escalating to Expert Agent for advanced analysis...")

                    # Log escalation
                    shared_memory.log_handoff(
                        st.session_state.current_agent,
                        "expert",
                        parsed["escalate_reason"] or "Complex issue requiring expert analysis",
                        parsed["escalate_context"] or "Escalated from specialist agent"
                    )

                    # Switch to expert
                    st.session_state.current_agent = "expert"
                    st.session_state.escalation_count += 1

                    # Get expert response
                    expert_thread = st.session_state.agent_threads.get("expert")
                    expert_response, new_thread = run_agent_sync(
                        user_input, "expert", expert_thread
                    )
                    st.session_state.agent_threads["expert"] = new_thread

                    st.markdown(expert_response)
                    st.session_state.messages.append({"role": "assistant", "content": expert_response})
                else:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # Normal response
            else:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Enhanced Sidebar Controls
with st.sidebar:
    st.header("🎛️ System Controls")

    # UI Controls
    st.subheader("Display Settings")
    st.session_state.show_status_updates = st.checkbox(
        "Show status updates",
        value=st.session_state.show_status_updates,
        help="Toggle visibility of system status messages during conversations"
    )

    # Agent switching
    st.subheader("Agent Selection")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Router", use_container_width=True):
            st.session_state.current_agent = "router"
            st.rerun()

        if st.button("👤 Individual", use_container_width=True):
            st.session_state.current_agent = "individual"
            st.session_state.user_type = "individual"
            st.rerun()

    with col2:
        if st.button("🏢 Developer", use_container_width=True):
            st.session_state.current_agent = "developer"
            st.session_state.user_type = "developer"
            st.rerun()

        if st.button("⚖️ Expert", use_container_width=True):
            st.session_state.current_agent = "expert"
            st.rerun()

    # System actions
    st.subheader("System Actions")
    if st.button("🗑️ Clear All", use_container_width=True):
        for key in ["messages", "agent_threads", "current_agent", "user_type", "shared_memory"]:
            if key in st.session_state:
                if key == "current_agent":
                    st.session_state[key] = "router"
                else:
                    del st.session_state[key]
        st.rerun()

    # System Status
    st.subheader("📊 System Status")

    if VECTOR_STORE_ID:
        st.success("📚 Vector Store Connected")
    else:
        st.error("📚 Vector Store Not Connected")

    if BRAVE_API_KEY:
        st.success("🔍 Web Search Enabled")
    else:
        st.warning("🔍 Web Search Disabled")

    if st.session_state.user_type:
        st.write(f"**User Type:** {st.session_state.user_type.title()}")

    if st.session_state.escalation_count > 0:
        st.write(f"**Escalations:** {st.session_state.escalation_count}")

    # Active threads
    active_threads = len([t for t in st.session_state.agent_threads.values() if t])
    st.write(f"**Active Threads:** {active_threads}")

    # Memory usage
    if "shared_memory" in st.session_state:
        memory = st.session_state.shared_memory
        st.write(f"**Context Items:** {len(memory.get('conversation_context', []))}")
        st.write(f"**Handoffs:** {len(memory.get('agent_handoffs', []))}")

        # Show recent errors if any
        errors = memory.get('error_log', [])
        if errors:
            st.warning(f"⚠️ {len(errors)} error(s) logged")

    # Architecture info
    with st.expander("🏗️ System Architecture"):
        st.markdown("""
        **Pattern:** Hierarchical Multi-Agent

        **Agents:**
        - 🔄 Router: Entry point & orchestration
        - 👤 Individual: Personal transport needs  
        - 🏢 Developer: Commercial projects
        - ⚖️ Expert: Complex policy & legal issues

        **Shared Memory:**
        - User profiles & classification
        - Conversation context & history
        - Agent handoff tracking
        - Error logging & recovery
        """)

    # Vector store setup help
    with st.expander("📚 Vector Store Setup"):
        st.markdown("""
        **To enable policy document access:**

        1. Go to OpenAI Platform → Storage → Vector Stores
        2. Find your transport policy vector store
        3. Copy the Vector Store ID (starts with `vs_`)
        4. Add to your `.env` file:
           ```
           VECTOR_STORE_ID=vs_your_id_here
           ```
        5. Restart the application

        **Current Status:**
        """)
        if VECTOR_STORE_ID:
            st.success(f"✅ Connected: {VECTOR_STORE_ID}")
        else:
            st.error("❌ Not connected - add VECTOR_STORE_ID to .env file")

    # Web search setup help
    with st.expander("🔍 Web Search Setup"):
        st.markdown("""
        **To enable web search fallback:**

        1. Get a Brave Search API key from https://brave.com/search/api/
        2. Add to your `.env` file:
           ```
           BRAVE_API_KEY=your_brave_api_key_here
           ```
        3. Restart the application

        **How it works:**
        - Primary source: Your policy documents (vector store)
        - Fallback: Web search when documents lack information
        - Clearly labeled web sources with verification notes

        **Current Status:**
        """)
        if BRAVE_API_KEY:
            st.success("✅ Web search enabled")
        else:
            st.warning("⚠️ Web search disabled - add BRAVE_API_KEY to .env file")

    # Debug information
    if st.checkbox("🔧 Debug Mode"):
        st.subheader("Debug Information")

        # Environment variables
        with st.expander("Environment Variables"):
            env_vars = {
                "OPENAI_API_KEY": "✅ Set" if OPENAI_API_KEY else "❌ Missing",
                "VECTOR_STORE_ID": VECTOR_STORE_ID if VECTOR_STORE_ID else "❌ Missing",
                "BRAVE_API_KEY": "✅ Set" if BRAVE_API_KEY else "❌ Missing",
                "ROUTER_ASSISTANT_ID": get_env_var("ROUTER_ASSISTANT_ID") if get_env_var(
                    "ROUTER_ASSISTANT_ID") else "❌ Missing",
                "INDIVIDUAL_ASSISTANT_ID": get_env_var("INDIVIDUAL_ASSISTANT_ID") if get_env_var(
                    "INDIVIDUAL_ASSISTANT_ID") else "❌ Missing",
                "DEVELOPER_ASSISTANT_ID": get_env_var("DEVELOPER_ASSISTANT_ID") if get_env_var(
                    "DEVELOPER_ASSISTANT_ID") else "❌ Missing",
                "EXPERT_ASSISTANT_ID": get_env_var("EXPERT_ASSISTANT_ID") if get_env_var(
                    "EXPERT_ASSISTANT_ID") else "❌ Missing"
            }

            for var, value in env_vars.items():
                if value.startswith("✅"):
                    st.success(f"{var}: {value}")
                elif value.startswith("❌"):
                    st.error(f"{var}: {value}")
                else:
                    st.info(f"{var}: {value}")

        if "shared_memory" in st.session_state:
            memory = st.session_state.shared_memory

            with st.expander("User Profile"):
                st.json(memory.get("user_profile", {}))

            with st.expander("Recent Handoffs"):
                handoffs = memory.get("agent_handoffs", [])[-3:]
                for handoff in handoffs:
                    st.write(f"**{handoff['from']} → {handoff['to']}**")
                    st.write(f"Reason: {handoff['reason']}")
                    st.write(f"Time: {handoff['timestamp']}")
                    st.write("---")

            with st.expander("Error Log"):
                errors = memory.get("error_log", [])[-5:]
                for error in errors:
                    st.error(f"**{error['type']}:** {error['message']}")
                    st.write(f"Agent: {error.get('agent', 'Unknown')}")
                    st.write(f"Time: {error['timestamp']}")
                    st.write("---")

# Footer
st.markdown("---")
st.markdown("*Transport Policy Advisor - Powered by OpenAI Assistants API*")