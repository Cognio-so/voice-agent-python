import os
import logging
import time
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WebSearchState(TypedDict):
    """State for the web search graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    search_results: Optional[List[Dict[str, Any]]]

class TavilyWebSearchTool:
    """Reusable Tavily web search tool for LangGraph."""
    
    def __init__(
        self, 
        max_results: int = 1,
        api_key: Optional[str] = None,
        search_depth: str = "basic",
        topic: str = "general",
        include_raw_content: bool = False,
        time_range: Optional[str] = None
    ):
        """
        Initialize the Tavily web search tool.
        
        Args:
            max_results: Maximum number of search results to return
            api_key: Tavily API key (defaults to TAVILY_API_KEY env variable)
            search_depth: Depth of search ("basic" or "advanced")
            topic: Category of search ("general", "news", or "finance")
            include_raw_content: Include cleaned HTML content
            time_range: Time range for results ("day", "week", "month", "year")
        """
        self.max_results = max_results
        self.api_key = api_key
        
        # Set API key in environment if provided
        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key
        elif not os.getenv("TAVILY_API_KEY"):
            raise ValueError("Tavily API key is required. Set the TAVILY_API_KEY environment variable.")
        
        try:
            # Initialize the search tool with optimized parameters
            self.search_tool = TavilySearch(
                max_results=max_results,
                topic=topic,
                include_raw_content=include_raw_content,
                search_depth=search_depth,
                time_range=time_range
            )
            logger.info(f"TavilyWebSearchTool initialized with max_results={max_results}, search_depth={search_depth}")
        except Exception as e:
            logger.error(f"Failed to initialize TavilySearch: {str(e)}")
            raise
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a web search using Tavily.
        
        Args:
            query: The search query
            
        Returns:
            List of search result objects
        """
        try:
            # Log and time the search
            logger.info(f"Executing web search for query: {query}")
            start_time = time.time()
            
            # Using proper invoke format for the new TavilySearch
            search_results = self.search_tool.invoke({"query": query})
            
            elapsed = time.time() - start_time
            logger.info(f"Search completed in {elapsed:.2f}s for: {query}")
            
            return search_results
        except Exception as e:
            logger.error(f"Error executing web search: {str(e)}")
            # Return empty results instead of raising exception
            return {"results": [], "error": str(e)}
    
    def get_tool(self) -> StructuredTool:
        """
        Get the underlying LangChain StructuredTool.
        
        Returns:
            StructuredTool instance for use in chains
        """
        return self.search_tool
    
    def create_tool_node(self) -> ToolNode:
        """
        Create a LangGraph ToolNode for the search tool.
        
        Returns:
            ToolNode instance for use in LangGraph
        """
        return ToolNode(tools=[self.search_tool])
    
    def bind_to_llm(self, llm):
        """
        Bind the tool to an LLM.
        
        Args:
            llm: Language model instance
            
        Returns:
            LLM with tools binding
        """
        return llm.bind_tools([self.search_tool])

def get_llm(model_name: str, temperature: float = 0.5):
    """
    Get an LLM instance based on the model name.
    
    Args:
        model_name: Name of the LLM to use
        temperature: Temperature setting for the LLM
        
    Returns:
        LLM instance
    """
    try:
        if "gemini" in model_name.lower():
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("Google API key is required for Gemini models.")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite-preview-06-17",
                temperature=temperature,
                google_api_key=google_api_key
            )


        elif "gpt" in model_name.lower():
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for GPT models.")
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature,
                api_key=openai_api_key
            )
        
        elif "claude" in model_name.lower():
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required for Claude models.")
            return ChatAnthropic(
                model="claude-3-5-sonnet-20240620",
                temperature=temperature,
                api_key=anthropic_api_key
            )
        
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    except Exception as e:
        logger.error(f"Error initializing language model: {str(e)}")
        raise

def get_search_components(llm):
    """
    Get components needed for web search without creating graph nodes.
    
    Args:
        llm: Language model instance
        
    Returns:
        Dictionary with search tool and LLM with tools bound
    """
    search_tool = TavilyWebSearchTool(max_results=1)
    llm_with_tools = search_tool.bind_to_llm(llm)
    
    return {
        "search_tool": search_tool,
        "llm_with_tools": llm_with_tools
    }
