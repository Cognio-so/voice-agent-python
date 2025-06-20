import os
import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Annotated
from datetime import datetime
from dotenv import load_dotenv

# LiveKit imports
from livekit import agents
from livekit.agents import (
    AgentSession, 
    Agent, 
    RoomInputOptions, 
    function_tool, 
    RunContext,
    JobContext
)
from livekit.agents import stt, tts
from livekit.plugins import google, openai, noise_cancellation, silero, deepgram

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
# from langsmith.run_helpers import traceable

# Import our custom modules
from langgraph_websearcch import TavilyWebSearchTool, WebSearchState, get_llm, get_search_components
from deepinfra_openai_tts import DeepInfraOpenAITTS
from voice_agent import ReliableDeepgramSTT, ReliableTTS

# Add import for OpenAI Assistant integration
from gpt_assistant import OptimizedOpenAIAssistantTool, FileUploadTool, create_openai_assistant_tools, get_assistant_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add this import at the top of main_demo_langgraph.py
from openai import OpenAI

# Enhanced state for the React agent
class ReactAgentState(WebSearchState):
    """Enhanced state for React agent with voice capabilities."""
    messages: Annotated[List[BaseMessage], add_messages]
    search_results: Optional[List[Dict[str, Any]]]
    voice_input: Optional[str]
    voice_output: Optional[str]
    conversation_history: List[BaseMessage]
    current_tool_calls: List[Dict[str, Any]]
    agent_scratchpad: List[BaseMessage]
    file_data: Optional[Dict[str, Any]] = None  # For assistant file data
    assistant_thread_id: Optional[str] = None  # For OpenAI Assistant thread continuity
    
class LangGraphVoiceAgent:
    """
    LangGraph-powered voice agent with web search capabilities using React pattern.
    Integrates with LiveKit for voice functionality.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite-preview-06-17",
        temperature: float = 0.5,
        language: str = "multi",
        max_search_results: int = 1,
        deepinfra_model: str = "canopylabs/orpheus-3b-0.1-ft",
        deepinfra_voice: str = "tara",
        deepgram_model: str = "nova-2",
        google_model: str = "gemini-2.5-flash-lite-preview-06-17",
        assistant_id: Optional[str] = None,  # New parameter for OpenAI Assistant
    ):
        """Initialize the LangGraph voice agent."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_search_results = max_search_results
        self.language = language
        self.deepinfra_model = deepinfra_model
        self.deepinfra_voice = deepinfra_voice
        self.deepgram_model = deepgram_model
        self.google_model = google_model
        self.assistant_id = assistant_id or os.getenv("OPENAI_ASSISTANT_ID")
        
        # Initialize components
        self.search_tool = None
        self.llm = None
        self.graph = None
        self.session = None
        self.memory = MemorySaver()
        self.assistant_tools = None  # Store the OpenAI Assistant tools
        
        # Initialize the components
        self._initialize_components()
        
        logger.info("LangGraphVoiceAgent initialized successfully")
    
    def _initialize_components(self):
        """Initialize only the OpenAI Assistant tool."""
        try:
            # Initialize LLM
            self.llm = get_llm(self.model_name, self.temperature)
            
            # Use specific assistant ID from environment variable
            os.environ["OPENAI_ASSISTANT_ID"] = "asst_BsP2eMHSIkpyewYQQGevAsAv"
            self.assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
            
            # Initialize OpenAI Assistant tools only
            self.assistant_tools = create_openai_assistant_tools(assistant_id=self.assistant_id)
            
            # No search tool initialization or binding
            self.search_tool = None
            
            # Build simplified graph with only assistant tool
            self.graph = self._build_assistant_only_graph()
            
            logger.info(f"Assistant initialized with ID: {self.assistant_id}")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def _build_assistant_only_graph(self) -> StateGraph:
        """Build a React agent graph with only OpenAI Assistant capabilities."""
        # Create the graph
        graph_builder = StateGraph(ReactAgentState)
        
        # Get OpenAI Assistant tools
        assistant_tool = self.assistant_tools[0]  # The file search tool
        file_upload_tool = self.assistant_tools[1]  # The file upload tool
        
        # Simple prompt that only includes OpenAI Assistant capabilities
        react_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with document search capabilities.

Current date and time: {current_datetime}

IMPORTANT INSTRUCTIONS FOR TOOL USAGE:
1. For searching through uploaded documents and files, use the openai_assistant_file_search tool:
```json
{{"tool_calls": [
  {{
    "name": "openai_assistant_file_search", 
    "args": {{ 
      "query": "your document search query",
      "thread_id": "optional thread ID from previous interactions"
    }}
  }}
]}}
```

2. To upload a file for searching, use upload_file_to_assistant:
```json
{{"tool_calls": [
  {{
    "name": "upload_file_to_assistant", 
    "args": {{ "file_path": "path to file" }}
  }}
]}}
```

3. Keep voice responses under 100 words.
4. Be direct and clear in your responses.

Available tools:
- openai_assistant_file_search: Search through uploaded documents and files.
- upload_file_to_assistant: Upload a file for future searching.
"""),
            ("placeholder", "{messages}")
        ])
        
        # Define the agent node
        def agent_node(state: ReactAgentState) -> Dict:
            """The main reasoning node of the React agent."""
            try:
                # Format the prompt with current datetime
                formatted_prompt = react_prompt.format_messages(
                    messages=state["messages"],
                    current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                # Get response from LLM
                response = self.llm.invoke(formatted_prompt)  # Use LLM directly without tools
                
                return {
                    "messages": [response],
                    "agent_scratchpad": state.get("agent_scratchpad", []) + [response]
                }
                
            except Exception as e:
                logger.error(f"Error in agent node: {str(e)}")
                error_response = AIMessage(
                    content="I apologize, but I encountered an error processing your request. Please try again."
                )
                return {
                    "messages": [error_response],
                    "agent_scratchpad": state.get("agent_scratchpad", []) + [error_response]
                }
        
        # Define the tool execution node with only OpenAI Assistant tools
        def tool_node(state: ReactAgentState) -> Dict:
            """Execute assistant tools and return results."""
            try:
                logger.info("Tool node called, executing assistant tool...")
                
                # Use only assistant tools
                tool_executor = ToolNode(tools=[
                    assistant_tool,
                    file_upload_tool
                ])
                
                result = tool_executor.invoke(state)
                
                # Extract thread ID for OpenAI Assistant
                thread_id = state.get("assistant_thread_id")
                for msg in result.get("messages", []):
                    if isinstance(msg, ToolMessage) and "thread_id" in str(msg.content):
                        try:
                            import json
                            content_dict = json.loads(msg.content)
                            if "thread_id" in content_dict:
                                thread_id = content_dict["thread_id"]
                        except:
                            pass
                
                return {
                    "messages": result.get("messages", []),
                    "assistant_thread_id": thread_id
                }
                
            except Exception as e:
                logger.error(f"Error in tool node: {str(e)}")
                error_message = ToolMessage(
                    content=f"Tool execution failed: {str(e)}",
                    tool_call_id="error"
                )
                return {
                    "messages": [error_message]
                }
        
        # Add nodes to the graph
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tool_node)
        
        # Use the prebuilt tools_condition
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,  # Use the prebuilt condition
            {
                "tools": "tools",
                END: END
            }
        )
        
        # Tools always return to agent
        graph_builder.add_edge("tools", "agent")
        
        # Set entry point
        graph_builder.add_edge(START, "agent")
        
        # Compile with memory
        return graph_builder.compile(checkpointer=self.memory)
    
    def _extract_search_results(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Extract search results from tool messages."""
        search_results = []
        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    # Parse the tool result
                    import json
                    if message.content:
                        result = json.loads(message.content)
                        if isinstance(result, list):
                            search_results.extend(result)
                        else:
                            search_results.append(result)
                except:
                    pass
        return search_results
    
    async def initialize_voice_session(self) -> Optional[AgentSession]:
        """Initialize the voice session with LiveKit."""
        try:
            # Get API keys
            deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
            
            if not all([deepinfra_api_key, google_api_key, deepgram_api_key]):
                missing = []
                if not deepinfra_api_key: missing.append("DEEPINFRA_API_KEY")
                if not google_api_key: missing.append("GOOGLE_API_KEY")
                if not deepgram_api_key: missing.append("DEEPGRAM_API_KEY")
                raise ValueError(f"Missing API keys: {', '.join(missing)}")
            
            # Create the session
            self.session = AgentSession(
                vad=silero.VAD.load(),
                stt=ReliableDeepgramSTT(
                    model=self.deepgram_model,
                    api_key=deepgram_api_key,
                    max_retries=1,
                    language=self.language
                ),
                llm=google.LLM(
                    model=self.google_model,
                    api_key=google_api_key
                ),
                tts=ReliableTTS(
                    voice=self.deepinfra_voice,
                    model=self.deepinfra_model,
                    api_key=deepinfra_api_key,
                    max_retries=1
                )
            )
            
            logger.info("Voice session initialized successfully")
            return self.session
            
        except Exception as e:
            logger.error(f"Error initializing voice session: {str(e)}")
            return None
    
    # @traceable(name="process_voice_query")
    async def process_voice_query(
        self, 
        query: str, 
        thread_id: str = "default",
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> str:
        """Process a voice query through the React agent graph."""
        try:
            # Start timing
            start_time = time.time()
            
            # Prepare the state
            messages = conversation_history or []
            messages.append(HumanMessage(content=query))
            
            initial_state = {
                "messages": messages,
                "search_results": None,
                "voice_input": query,
                "voice_output": None,
                "conversation_history": messages,
                "current_tool_calls": [],
                "agent_scratchpad": []
            }
            
            # Process through the graph with thread management
            config = {"configurable": {"thread_id": thread_id}}
            
            # Use streaming for better responsiveness
            response_text = ""
            async for event in self.graph.astream(initial_state, config=config):
                if "messages" in event:
                    for message in event["messages"]:
                        if isinstance(message, AIMessage) and message.content:
                            response_text += message.content
            
            # If no response, use the last message
            if not response_text:
                final_state = await self.graph.ainvoke(initial_state, config=config)
                last_message = final_state.get("messages", [])[-1]
                if isinstance(last_message, AIMessage):
                    response_text = last_message.content
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds: '{query[:30]}...'")
            
            # Append timing info to response for voice (optional)
            response_with_time = f"{response_text}\n(Response time: {elapsed_time:.1f}s)"
            
            return response_with_time
            
        except Exception as e:
            logger.error(f"Error processing voice query: {str(e)}")
            return "I encountered an error while processing your request. Please try again."
    
    def create_livekit_function_tool(self):
        """Create a LiveKit function tool that uses our LangGraph agent."""
        
        @function_tool
        async def langgraph_web_search(
            context: RunContext,
            query: str,
        ) -> str:
            """Search the web using LangGraph React agent with voice optimization."""
            try:
                start_time = time.time()
                
                # Process through our React agent
                response = await self.process_voice_query(
                    query=query,
                    thread_id=f"voice_session_{id(context)}"
                )
                
                # Optimize response for voice (limit length)
                if len(response) > 50:
                    sentences = response.split('. ')
                    truncated = '. '.join(sentences[:2])
                    if len(truncated) < len(response):
                        response = truncated + '.'
                
                elapsed = time.time() - start_time
                logger.info(f"LangGraph web search completed in {elapsed:.2f}s")
                
                # For voice response, we can add timing at the end
                response = f"{response} (Answered in {elapsed:.1f} seconds)"
                
                return response
                
            except Exception as e:
                logger.error(f"Error in LangGraph web search: {str(e)}")
                return "I'm having trouble searching for that information right now. Please try again."
        
        return langgraph_web_search

    def create_openai_assistant_function_tool(self):
        """Create a LiveKit function tool that uses OpenAI Assistant."""
        
        # Create a persistent thread ID for this session
        persistent_thread_id = None
        
        @function_tool
        async def openai_assistant_search(
            context: RunContext,
            query: str,
        ) -> str:
            """Search through documents using OpenAI Assistant."""
            try:
                nonlocal persistent_thread_id
                start_time = time.time()
                
                # Get the OpenAI Assistant tool
                assistant_tool = self.assistant_tools[0]
                
                # Use direct OpenAI API calls for better thread management
                try:
                    # Import is already at the file top
                    client = OpenAI()
                    assistant_id = self.assistant_id
                    
                    # Create thread if it doesn't exist
                    if persistent_thread_id is None:
                        logger.info("Creating new thread for OpenAI Assistant")
                        thread = client.beta.threads.create()
                        persistent_thread_id = thread.id
                        logger.info(f"Created new thread ID: {persistent_thread_id}")
                    
                    logger.info(f"Using thread ID: {persistent_thread_id}")
                    
                    # Add message to thread
                    client.beta.threads.messages.create(
                        thread_id=persistent_thread_id,
                        role="user",
                        content=query
                    )
                    
                    # Run the assistant
                    run = client.beta.threads.runs.create(
                        thread_id=persistent_thread_id,
                        assistant_id=assistant_id
                    )
                    
                    # Wait for completion
                    logger.info("Waiting for OpenAI Assistant to process the query...")
                    while run.status not in ["completed", "failed"]:
                        await asyncio.sleep(0.5)
                        run = client.beta.threads.runs.retrieve(
                            thread_id=persistent_thread_id,
                            run_id=run.id
                        )
                        
                    if run.status == "failed":
                        raise Exception(f"Assistant run failed: {run.last_error}")
                        
                    # Get the latest assistant message
                    messages = client.beta.threads.messages.list(thread_id=persistent_thread_id)
                    response = next((m.content[0].text.value for m in messages.data if m.role == "assistant"), 
                                   "No response from assistant")
                    
                    logger.info(f"Got response from OpenAI Assistant using thread: {persistent_thread_id}")
                    
                except Exception as e:
                    logger.error(f"Direct API call failed: {str(e)}")
                    # Don't use the tool as fallback since it has the same thread issue
                    response = f"I encountered an error accessing the assistant: {str(e)}"
                
                # Optimize response for voice (limit length)
                if len(response) > 100:
                    sentences = response.split('. ')
                    truncated = '. '.join(sentences[:2])
                    if len(truncated) < len(response):
                        response = truncated + '.'
                
                elapsed = time.time() - start_time
                logger.info(f"OpenAI Assistant search completed in {elapsed:.2f}s")
                
                return response
                
            except Exception as e:
                logger.error(f"Error in OpenAI Assistant search: {str(e)}")
                return "I'm having trouble searching through the documents right now. Please try again."
        
        return openai_assistant_search


class LangGraphVoiceAssistant:
    """
    Complete voice assistant using LangGraph for reasoning and LiveKit for voice.
    """
    
    def __init__(self, **kwargs):
        """Initialize the voice assistant."""
        self.langgraph_agent = LangGraphVoiceAgent(**kwargs)
        self.session = None
        self.current_agent = None
        
    async def initialize_session(self) -> bool:
        """Initialize the voice session."""
        self.session = await self.langgraph_agent.initialize_voice_session()
        return self.session is not None
    
    def _create_livekit_agent(self) -> Agent:
        """Create the LiveKit agent with OpenAI Assistant integration only."""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get the OpenAI Assistant tool only
        assistant_tool = self.langgraph_agent.create_openai_assistant_function_tool()
        
        # Create the agent with only the assistant tool
        agent = Agent(
            instructions=f"""You are an intelligent voice assistant powered by OpenAI Assistant with document search capabilities.

Current date and time: {current_datetime}

CORE CAPABILITIES:
- Document search through uploaded files
- Intelligent reasoning and analysis
- Conversational voice interactions

VOICE INTERACTION GUIDELINES:
- Keep responses concise and natural (under 100 words)
- Use document search for questions about uploaded files
- Provide direct, helpful answers
- Speak conversationally and naturally
- Acknowledge when you're searching for information

TOOL USAGE:
- Use openai_assistant_search for searching through uploaded documents
- Always provide accurate information from documents
- Be transparent about your search process

Remember: This is a voice conversation, so prioritize clarity and brevity.""",
            tools=[assistant_tool]  # Only use assistant tool
        )
        
        return agent
    
    async def start(self, ctx: JobContext) -> None:
        """Start the voice assistant."""
        try:
            # Initialize session
            if not await self.initialize_session():
                logger.error("Failed to initialize voice session")
                return
            
            # Create the agent
            self.current_agent = self._create_livekit_agent()
            
            # Start the session
            await self.session.start(
                room=ctx.room,
                agent=self.current_agent,
                room_input_options=RoomInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),
                ),
            )
            
            # Connect to the room
            await ctx.connect()
            
            # Generate initial greeting
            await self.session.generate_reply(
                instructions="Greet the user warmly and introduce yourself as an AI assistant with web search capabilities. Keep it brief and friendly."
            )
            
            logger.info("✅ LangGraph Voice Assistant started successfully!")
            
        except Exception as e:
            logger.error(f"Error starting voice assistant: {str(e)}")
            raise
    
    async def run(self, ctx: JobContext) -> None:
        """Run the voice assistant."""
        await self.start(ctx)
        
        try:
            # Keep the session running
            logger.info("🎤 LangGraph Voice Assistant is now listening...")
            
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Session terminated by user. Goodbye!")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")


# Main entrypoint for LiveKit
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LangGraph voice assistant."""
    try:
        # Set the assistant ID directly in environment
        os.environ["OPENAI_ASSISTANT_ID"] = os.getenv("OPENAI_ASSISTANT_ID")
        
        # Create and run the assistant
        assistant = LangGraphVoiceAssistant(
            model_name="gpt-4o-mini",  # Can be changed to other models
            temperature=0.5,
            max_search_results=1,
            google_model="gemini-2.5-flash-lite-preview-06-17",
            assistant_id=os.getenv("OPENAI_ASSISTANT_ID")  # Now we're sure this is set
        )
        
        await assistant.run(ctx)
        
    except Exception as e:
        logger.error(f"Critical error in entrypoint: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    # Run LiveKit agent
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
