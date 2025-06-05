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

# Suppress OpenAI deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openai")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        "model": "gpt-4o"
    },
    "individual": {
        "name": "Individual Transport Advisor",
        "role": "Personal Transport Specialist",
        "instructions": """You are a friendly transport advisor helping individuals with personal needs:

EXPERTISE:
- Public transport accessibility and routes
- Walking/cycling infrastructure and safety
- Personal mobility and accessibility needs
- Transport costs, ticketing, and entitlements
- Local transport policies affecting daily life
- Rights as a transport user

APPROACH:
- Use conversational, empathetic tone
- Ask about specific location and circumstances
- Provide practical, actionable advice
- Focus on immediate, personal solutions

ESCALATION: If you encounter complex legal/regulatory issues, respond with:
ESCALATE: EXPERT
REASON: [Why escalation needed]
CONTEXT: [Summary for expert agent]""",
        "model": "gpt-4o",
        "tools": [{"type": "file_search"}]
    },
    "developer": {
        "name": "Commercial Developer Advisor",
        "role": "Commercial Development Specialist",
        "instructions": """You are a professional transport consultant for commercial development:

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
- Use professional, technical language
- Focus on regulatory compliance and planning requirements
- Ask about project scale, location, and development type
- Provide commercially-focused solutions

ESCALATION: For complex legal disputes or technical regulations:
ESCALATE: EXPERT
REASON: [Why escalation needed]
CONTEXT: [Summary for expert agent]""",
        "model": "gpt-4o",
        "tools": [{"type": "file_search"}]
    },
    "expert": {
        "name": "Transport Policy Expert",
        "role": "Advanced Policy & Legal Specialist",
        "instructions": """You are a senior transport policy expert handling complex issues:

EXPERTISE:
- Advanced regulatory interpretation
- Legal disputes and appeals processes
- Technical policy analysis
- Cross-jurisdictional transport law
- Planning inquiries and tribunals
- Complex infrastructure assessments
- Policy development and reform

APPROACH:
- Provide authoritative, detailed analysis
- Reference specific legislation and case law
- Explain complex procedures step-by-step
- Offer strategic advice for difficult situations
- Use precise legal and technical terminology

You handle the most complex cases that other agents cannot resolve.""",
        "model": "gpt-4o",
        "tools": [{"type": "file_search"}]
    }
}


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
        "escalation_count": 0
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

# Streamlit UI Setup
st.set_page_config(page_title="Multi-Agent Transport Policy Advisor", layout="wide")
st.title("🚦 Multi-Agent Transport Policy Advisor")
st.markdown("Advanced AI system with specialized agents for different transport policy needs")

# Agent Status Display
agent_info = {
    "router": {"icon": "🔄", "name": "Router Agent", "role": "Classifying and routing queries"},
    "individual": {"icon": "👤", "name": "Individual Advisor", "role": "Personal transport guidance"},
    "developer": {"icon": "🏢", "name": "Developer Advisor", "role": "Commercial development support"},
    "expert": {"icon": "⚖️", "name": "Expert Advisor", "role": "Complex policy & legal analysis"}
}

current_info = agent_info.get(st.session_state.current_agent, agent_info["router"])
st.info(f"{current_info['icon']} **{current_info['name']}** - {current_info['role']}")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Agent Management Functions
async def create_or_get_assistant(agent_type: str) -> str:
    """Create or retrieve assistant for given agent type with error handling"""
    try:
        config = AGENT_CONFIG[agent_type]

        # Try environment variable first
        env_var = f"{agent_type.upper()}_ASSISTANT_ID"
        existing_id = os.getenv(env_var)

        if existing_id:
            try:
                assistant = await client.beta.assistants.retrieve(existing_id)
                return assistant.id
            except Exception as e:
                shared_memory.log_error("assistant_retrieval", f"Failed to retrieve {agent_type}: {str(e)}")

        # Create new assistant
        assistant_params = {
            "name": config["name"],
            "instructions": config["instructions"],
            "model": config["model"]
        }

        if "tools" in config:
            assistant_params["tools"] = config["tools"]

        assistant = await client.beta.assistants.create(**assistant_params)

        st.success(f"✅ Created {config['name']}: {assistant.id}")
        st.info(f"💡 Add {env_var}={assistant.id} to your .env file")

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
    """Call specific agent with comprehensive error handling"""
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
            enhanced_input = f"Context: {context}\n\nUser Query: {user_input}"
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


# Main Chat Interface
if user_input := st.chat_input("Ask your transport policy question..."):
    # Validate setup
    if not OPENAI_API_KEY:
        st.error("🔑 OpenAI API key not found. Please check your .env file.")
        st.stop()

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

                    # Show classification
                    st.success(f"✅ Classified as: {classification.title()}")
                    if parsed["reason"]:
                        st.info(f"📝 Reason: {parsed['reason']}")

                    # Get response from specialist
                    specialist_thread = st.session_state.agent_threads.get(classification)
                    specialist_response, new_thread = run_agent_sync(
                        user_input, classification, specialist_thread
                    )
                    st.session_state.agent_threads[classification] = new_thread

                    st.markdown(specialist_response)
                    st.session_state.messages.append({"role": "assistant", "content": specialist_response})
                else:
                    # Continue with router
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # Handle escalation requests
            elif parsed["escalate"]:
                escalate_to = parsed["escalate"].lower()
                if escalate_to == "expert":
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

    # Debug information
    if st.checkbox("🔧 Debug Mode"):
        st.subheader("Debug Information")

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
st.markdown("*Multi-Agent Transport Policy Advisor - Powered by OpenAI Assistants API*")