import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from groq import Groq
import google.generativeai as genai
import time
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Multi-Agent Debate System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .agent-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .agent-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .agent-content {
        font-size: 1rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .stButton > button {
        width: 100%;
    }
    .chat-history {
        max-height: 300px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-bottom: 1rem;
    }
    .history-item {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        background-color: white;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

class ChatHistory:
    def __init__(self, max_entries: int = 10):
        self.max_entries = max_entries
        self.history_file = "chat_history.json"
        self.load_history()

    def load_history(self):
        if not hasattr(st.session_state, '_chat_history'):
            st.session_state._chat_history = []
            if os.path.exists(self.history_file):
                try:
                    with open(self.history_file, 'r') as f:
                        st.session_state._chat_history = json.load(f)
                except:
                    st.session_state._chat_history = []

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(st.session_state._chat_history[-self.max_entries:], f)

    def add_entry(self, question: str, final_answer: str):
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'question': question,
            'final_answer': final_answer
        }
        st.session_state._chat_history.append(entry)
        if len(st.session_state._chat_history) > self.max_entries:
            st.session_state._chat_history.pop(0)
        self.save_history()

    def get_history(self):
        return st.session_state._chat_history

    def display_history(self):
        if not st.session_state._chat_history:
            st.info("No chat history available yet.")
            return

        st.write("### 📚 Previous Discussions")
        with st.container():
            for entry in reversed(st.session_state._chat_history):
                st.markdown(f"""
                <div class="history-item">
                    <div class="timestamp">{entry['timestamp']}</div>
                    <strong>Q: {entry['question']}</strong><br>
                    <em>Final Answer:</em> {entry['final_answer'][:200]}...
                </div>
                """, unsafe_allow_html=True)

class LLMClient:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env file")
            self.client = Groq(api_key=api_key)
            self.model = "deepseek-r1-distill-llama-70b"
        elif provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-pro')
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            if self.provider == "groq":
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=kwargs.get('temperature', 0.7),
                    max_tokens=4096,
                    top_p=0.95,
                )
                return completion.choices[0].message.content
            else:  # gemini
                prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=kwargs.get('temperature', 0.7)
                    )
                )
                return response.text
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Error generating response. Please try again."

class Agent:
    SYSTEM_PROMPTS = {
        "proposer": """You are the Proposer agent in a multi-agent debate system. Your role is to:
1. Analyze the given question thoroughly
2. Consider the context and chat history if provided
3. Generate a comprehensive initial solution
4. Structure your response as follows:
   - Context Analysis:
   - Key Points:
   - Detailed Solution:
   - Potential Implications:
   - References to Past Discussions (if relevant):""",
        
        "critic": """You are the Critic agent in a multi-agent debate system. Your role is to:
1. Analyze the previous responses carefully
2. Consider historical context and past discussions
3. Identify potential flaws and improvements
4. Structure your response as follows:
   - Context Understanding:
   - Strengths Identified:
   - Critical Analysis:
   - Missing Considerations:
   - Suggested Improvements:
   - Historical Pattern Analysis (if relevant):""",
        
        "refiner": """You are the Refiner agent in a multi-agent debate system. Your role is to:
1. Consider the initial proposal and criticisms
2. Analyze historical context and patterns
3. Create an improved solution
4. Structure your response as follows:
   - Context Integration:
   - Addressed Concerns:
   - Refined Solution:
   - Implementation Steps:
   - Connection to Past Solutions (if relevant):
   - Risk Mitigation:""",
        
        "synthesizer": """You are the Synthesizer agent in a multi-agent debate system. Your role is to:
1. Analyze the entire discussion thread
2. Consider historical patterns and insights
3. Create a comprehensive final answer
4. Structure your response as follows:
   - Discussion Summary:
   - Key Insights:
   - Synthesized Solution:
   - Implementation Strategy:
   - Connection to Previous Discussions:
   - Future Considerations:"""
    }

    def __init__(self, name: str, llm_client: LLMClient, role: str, color: str):
        self.name = name
        self.llm_client = llm_client
        self.role = self.SYSTEM_PROMPTS[role.lower()]
        self.color = color

    def generate_response(self, context: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": self.prepare_context(context)}
        ]
        
        with st.spinner(f"{self.name} is thinking..."):
            response = self.llm_client.generate_response(messages)
            return response

    def prepare_context(self, context: Dict[str, Any]) -> str:
        # Start with the current question
        content = f"Question: {context.get('question', '')}\n\n"
        
        # Calculate approximate token lengths (rough estimate: 4 chars = 1 token)
        max_tokens = 2000  # Leave room for response
        current_tokens = len(content) // 4
        
        # Add current discussion context (prioritize recent messages)
        if 'discussion' in context and context['discussion']:
            content += "Current Discussion Thread:\n"
            # Only include the last 2 messages to keep context manageable
            for entry in context['discussion'][-2:]:
                message = f"\n{entry['agent']}:\n{entry['response'][:800]}\n"  # Truncate long messages
                if (current_tokens + len(message) // 4) < max_tokens:
                    content += message
                    current_tokens += len(message) // 4
                else:
                    break
        
        # Add relevant chat history (if there's token budget left)
        if 'chat_history' in context and context['chat_history']:
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 200:  # Only add history if we have enough token budget
                content += "\nRelevant Historical Context:\n"
                # Only take the last relevant discussion
                latest_entry = context['chat_history'][-1]
                history_content = (
                    f"Previous Discussion ({latest_entry['timestamp']}):\n"
                    f"Q: {latest_entry['question']}\n"
                    f"A: {latest_entry['final_answer'][:300]}...\n"  # Truncate long answers
                )
                if (current_tokens + len(history_content) // 4) < max_tokens:
                    content += history_content
        
        return content

class DebateSystem:
    def __init__(self):
        try:
            groq_client = LLMClient("groq")
            gemini_client = LLMClient("gemini")
            
            self.agents = {
                "proposer": Agent(
                    "Proposer",
                    groq_client,
                    "proposer",
                    "#FF9800"  # Orange
                ),
                "critic": Agent(
                    "Critic",
                    gemini_client,
                    "critic",
                    "#F44336"  # Red
                ),
                "refiner": Agent(
                    "Refiner",
                    groq_client,
                    "refiner",
                    "#2196F3"  # Blue
                ),
                "synthesizer": Agent(
                    "Synthesizer",
                    gemini_client,
                    "synthesizer",
                    "#4CAF50"  # Green
                )
            }
            
            self.discussion_history: List[Dict[str, str]] = []
            
        except Exception as e:
            st.error(f"Error initializing debate system: {str(e)}")
            st.stop()

    def display_response(self, agent_name: str, response: str, color: str):
        st.markdown(f"""
        <div class="agent-box" style="background-color: {color}20;">
            <div class="agent-header" style="color: {color};">
                {agent_name}
            </div>
            <div class="agent-content">
                {response}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def debate(self, question: str, max_iterations: int = 2) -> None:
        context = {
            "question": question,
            "discussion": [],
            "chat_history": st.session_state._chat_history
        }

        try:
            with st.container():
                st.write("### 🎯 Question")
                st.info(question)

                # 1. Get initial proposal
                st.write("### 💡 Initial Proposal")
                proposal = self.agents["proposer"].generate_response(context)
                self.discussion_history = [{"agent": "Proposer", "response": proposal}]
                self.display_response("Proposer", proposal, self.agents["proposer"].color)

                for i in range(max_iterations):
                    st.write(f"### 🔄 Iteration {i+1}/{max_iterations}")
                    context["discussion"] = self.discussion_history
                    
                    # 2. Get criticism
                    criticism = self.agents["critic"].generate_response(context)
                    self.discussion_history.append({"agent": "Critic", "response": criticism})
                    self.display_response("Critic", criticism, self.agents["critic"].color)

                    # 3. Get refined solution
                    context["discussion"] = self.discussion_history[-2:]
                    refinement = self.agents["refiner"].generate_response(context)
                    self.discussion_history.append({"agent": "Refiner", "response": refinement})
                    self.display_response("Refiner", refinement, self.agents["refiner"].color)

                # 4. Get final synthesis
                st.write("### ✨ Final Synthesis")
                context["discussion"] = [
                    self.discussion_history[0],
                    self.discussion_history[-1]
                ]
                final_answer = self.agents["synthesizer"].generate_response(context)
                self.discussion_history.append({"agent": "Synthesizer", "response": final_answer})
                self.display_response("Synthesizer", final_answer, self.agents["synthesizer"].color)
                
                # Save to chat history
                st.session_state.chat_history.add_entry(question, final_answer[:1000])
            
        except Exception as e:
            st.error(f"Error during debate: {str(e)}")

def main():
    st.title("🤖 Multi-Agent Debate System")
    st.markdown("---")

    # Initialize debate system and chat history
    if 'debate_system' not in st.session_state:
        st.session_state.debate_system = DebateSystem()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatHistory()

    # Display chat history in sidebar
    with st.sidebar:
        st.title("🔍 About")
        st.info(
            """
            The Multi-Agent Debate System combines the power of Groq and Gemini AI models 
            to provide comprehensive analysis of complex questions. Each agent has a specific 
            role and follows a structured template to ensure clear and organized responses.
            
            **Features:**
            - Structured responses
            - Multiple debate rounds
            - Chat history
            - Context awareness
            - Real-time updates
            """
        )
        st.session_state.chat_history.display_history()

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        This system uses multiple AI agents to analyze and discuss your question:
        
        1. 🟧 **Proposer**: Generates structured initial solutions with context awareness
        2. 🟥 **Critic**: Identifies potential flaws and improvements
        3. 🟦 **Refiner**: Improves the solution based on criticism and history
        4. 🟩 **Synthesizer**: Creates a comprehensive final synthesis
        
        Each agent follows a specific template and considers both the current discussion
        and historical context to provide more informed responses.
        """)

    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="Type your question here...",
            help="Ask any question and watch the agents debate!"
        )

    with col2:
        num_iterations = st.number_input(
            "Debate Rounds",
            min_value=1,
            max_value=3,
            value=2,
            help="Number of debate rounds between agents"
        )

    if st.button("🚀 Start Debate", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Please enter a question first.")
            return
        
        start_time = time.time()
        context = {
            "question": question,
            "discussion": [],
            "chat_history": st.session_state.chat_history.get_history()
        }
        st.session_state.debate_system.debate(question, num_iterations)
        end_time = time.time()
        
        st.success(f"✅ Debate completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
