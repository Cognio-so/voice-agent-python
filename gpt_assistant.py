import os
import asyncio
import logging
import time
import json
from typing import Optional, List, Dict, Any, Union, Type
from datetime import datetime
from pathlib import Path

# OpenAI imports
import openai
from openai import OpenAI
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run, Message

# LangChain imports for tool integration
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Use the official LangChain OpenAI Assistant integration
from langchain.agents.openai_assistant.base import OpenAIAssistantRunnable

# LangSmith for enhanced monitoring (following user rules)
from langsmith import traceable

# Add this import near the top
import nest_asyncio
nest_asyncio.apply()  # This allows nested event loops when needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenAIAssistantError(Exception):
    """Custom exception for OpenAI Assistant operations."""
    pass


class FileUploadInput(BaseModel):
    """Input model for file uploads."""
    file_path: str = Field(description="Path to the file to upload")
    purpose: str = Field(default="assistants", description="Purpose of the file upload")


class AssistantQueryInput(BaseModel):
    """Input model for querying the assistant."""
    query: str = Field(description="The question or task to send to the assistant")
    thread_id: Optional[str] = Field(default=None, description="Optional thread ID to continue conversation")
    include_file_search: bool = Field(default=True, description="Whether to use file search capabilities")


# Raw helper function as recommended in the pattern
def ask_openai_assistant(user_input: str, assistant_id: str, thread_id: str = None, client: OpenAI = None):
    """
    Raw helper to talk to OpenAI Assistant API directly.
    Returns (assistant_reply, thread_id) for stateless operation.
    """
    if client is None:
        client = OpenAI()  # Uses OPENAI_API_KEY from environment
    
    if thread_id is None:
        thread_id = client.beta.threads.create().id

    # 1️⃣ Add the user message
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input,
    )

    # 2️⃣ Start a run
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    # 3️⃣ Poll until finished
    while run.status not in {"completed", "failed"}:
        time.sleep(0.4)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id,
        )
    
    if run.status == "failed":
        raise RuntimeError(f"Assistant run failed: {run.last_error}")

    # 4️⃣ Fetch the latest assistant message
    msgs = client.beta.threads.messages.list(thread_id=thread_id)
    answer = next(m.content[0].text.value for m in msgs.data if m.role == "assistant")
    return answer, thread_id


class OptimizedOpenAIAssistantManager:
    """
    Optimized OpenAI Assistant Manager that works with existing assistants.
    Uses the official LangChain OpenAIAssistantRunnable for better integration.
    """
    
    def __init__(
        self,
        assistant_id: Optional[str] = None,
        api_key: Optional[str] = None,
        create_new_assistant: bool = False,
        assistant_name: str = "rag_voicebot",
        assistant_instructions: str = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3
    ):
        """
        Initialize the Optimized OpenAI Assistant Manager.
        
        Args:
            assistant_id: Existing assistant ID (preferred approach)
            api_key: OpenAI API key (defaults to environment variable)
            create_new_assistant: Whether to create a new assistant if none provided
            assistant_name: Name for new assistant (if creating)
            assistant_instructions: Instructions for new assistant (if creating)
            model: OpenAI model to use
            max_retries: Maximum number of retries for operations
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.assistant_id = assistant_id
        self.max_retries = max_retries
        self.model = model
        
        # Default instructions for new assistants
        if assistant_instructions is None:
            self.assistant_instructions = """You are an intelligent AI assistant with file search capabilities integrated into a voice-enabled RAG system.

Your capabilities include:
1. Searching through uploaded documents and files to find relevant information
2. Answering questions based on file content with high accuracy
3. Providing detailed analysis of document content
4. Citing sources from the files when providing information

Guidelines for responses:
- Always cite specific documents or sections when referencing file content
- Provide accurate and comprehensive answers based on available information
- If information isn't found in the files, clearly state that
- Keep responses concise and voice-friendly (under 100 words when possible)
- Use clear, conversational language
- Prioritize the most relevant information first

For voice interactions, be natural and conversational while maintaining accuracy."""
        else:
            self.assistant_instructions = assistant_instructions
        
        # Initialize assistant runnable
        self.assistant_runnable = None
        self.assistant_info = None
        
        # Setup the assistant
        self._setup_assistant(create_new_assistant, assistant_name)
        
        logger.info(f"Optimized OpenAI Assistant Manager initialized with ID: {self.assistant_id}")
    
    def _setup_assistant(self, create_new: bool, assistant_name: str):
        """Setup the assistant - either use existing or create new."""
        try:
            if self.assistant_id:
                # Use existing assistant - directly get it using the API reference method
                logger.info(f"Retrieving existing assistant: {self.assistant_id}")
                try:
                    # Use the direct API method to get assistant
                    self.assistant_info = self.client.beta.assistants.retrieve(
                        assistant_id=self.assistant_id
                    )
                    
                    # Create the runnable wrapper
                    self.assistant_runnable = OpenAIAssistantRunnable(
                        assistant_id=self.assistant_id,
                        client=self.client
                    )
                    
                    logger.info(f"Retrieved assistant: {self.assistant_info.name}")
                    
                except Exception as e:
                    logger.error(f"Error retrieving assistant: {str(e)}")
                    if create_new:
                        logger.info("Creating new assistant as fallback...")
                        self._create_new_assistant(assistant_name)
                    else:
                        raise
                
            elif create_new:
                # Create new assistant
                self._create_new_assistant(assistant_name)
                
            else:
                raise ValueError("Either provide assistant_id or set create_new_assistant=True")
                
        except Exception as e:
            logger.error(f"Error setting up assistant: {str(e)}")
            raise OpenAIAssistantError(f"Failed to setup assistant: {str(e)}")

    def _create_new_assistant(self, assistant_name: str):
        """Helper method to create a new assistant."""
        logger.info("Creating new assistant...")
        assistant = self.client.beta.assistants.create(
            name=assistant_name,
            instructions=self.assistant_instructions,
            model=self.model,
            tools=[{"type": "file_search"}]  # Enable file search by default
        )
        
        self.assistant_id = assistant.id
        self.assistant_info = assistant
        
        self.assistant_runnable = OpenAIAssistantRunnable(
            assistant_id=self.assistant_id,
            client=self.client
        )
        
        logger.info(f"Created new assistant: {self.assistant_id}")
        print(f"🆕 Created new assistant: {self.assistant_id}")
        print("💡 Save this ID to OPENAI_ASSISTANT_ID environment variable for future use")
    
    @traceable(name="upload_file_optimized")
    def upload_file(self, file_path: str, purpose: str = "assistants") -> Dict[str, Any]:
        """
        Upload a file to OpenAI for use with the assistant.
        
        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file upload
        
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Upload the file
            with open(file_path, "rb") as file:
                uploaded_file = self.client.files.create(
                    file=file,
                    purpose=purpose
                )
            
            logger.info(f"File uploaded successfully: {file_path.name}")
            
            return {
                "file_id": uploaded_file.id,
                "filename": uploaded_file.filename,
                "bytes": uploaded_file.bytes,
                "purpose": uploaded_file.purpose
            }
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise OpenAIAssistantError(f"Failed to upload file: {str(e)}")
    
    @traceable(name="query_assistant_optimized")
    async def query_assistant(
        self, 
        query: str, 
        thread_id: Optional[str] = None,
        file_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        Query the assistant using the optimized approach.
        
        Args:
            query: Question or task for the assistant
            thread_id: Optional existing thread ID for conversation continuity
            file_ids: Optional file IDs to attach to the message
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()
            
            if not self.assistant_runnable:
                raise OpenAIAssistantError("Assistant not properly initialized")
            
            # Prepare the input
            input_data = {
                "content": query,
                "thread_id": thread_id
            }
            
            # If file_ids provided, we need to attach them to the message
            if file_ids:
                # Use the raw helper for file attachment support
                response, new_thread_id = ask_openai_assistant(
                    user_input=query,
                    assistant_id=self.assistant_id,
                    thread_id=thread_id,
                    client=self.client
                )
            else:
                # Use the LangChain runnable for simpler queries
                if thread_id:
                    input_data["thread_id"] = thread_id
                
                result = await self.assistant_runnable.ainvoke(input_data)
                response = result["output"]
                new_thread_id = result.get("thread_id", thread_id)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            result = {
                "response": response.strip(),
                "thread_id": new_thread_id,
                "response_time": response_time,
                "assistant_id": self.assistant_id,
                "model": self.model
            }
            
            logger.info(f"Query completed in {response_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error querying assistant: {str(e)}")
            raise OpenAIAssistantError(f"Failed to query assistant: {str(e)}")
    
    def get_assistant_info(self) -> Dict[str, Any]:
        """Get information about the current assistant."""
        if self.assistant_info:
            return {
                "id": self.assistant_info.id,
                "name": self.assistant_info.name,
                "model": self.assistant_info.model,
                "tools": [tool.type for tool in self.assistant_info.tools],
                "instructions": self.assistant_info.instructions[:100] + "..." if len(self.assistant_info.instructions) > 100 else self.assistant_info.instructions
            }
        return {}


class OptimizedOpenAIAssistantTool(BaseTool):
    """
    Optimized LangChain tool wrapper using OpenAIAssistantRunnable.
    This integrates seamlessly with LangGraph systems.
    """
    
    name: str = "openai_assistant_file_search"
    description: str = """Use this tool to search through uploaded documents and files for information.
    This assistant can:
    - Search through PDF, text, and document files
    - Answer questions based on file content
    - Provide detailed analysis of documents
    - Cite sources from the files
    
    Input should be a clear question or request about the file contents."""
    
    args_schema: Type[AssistantQueryInput] = AssistantQueryInput
    
    assistant_id: Optional[str] = None
    assistant_manager: Optional[Any] = None
    
    def __init__(self, assistant_id: str = None, **kwargs):
        super().__init__(**kwargs)
        
        self.assistant_id = assistant_id or os.getenv("OPENAI_ASSISTANT_ID")
        
        if not self.assistant_id:
            logger.warning("No assistant ID provided. Will create a new assistant.")
            self.assistant_manager = OptimizedOpenAIAssistantManager(
                create_new_assistant=True,
                assistant_name="LangGraph File Search Assistant"
            )
        else:
            logger.info(f"Using existing assistant: {self.assistant_id}")
            self.assistant_manager = OptimizedOpenAIAssistantManager(
                assistant_id=self.assistant_id
            )
        
        logger.info("Optimized OpenAI Assistant Tool initialized successfully")
    
    @traceable(name="optimized_assistant_tool_run")
    def _run(
        self, 
        query: str,
        thread_id: Optional[str] = None,
        include_file_search: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Execute the tool using the optimized approach.
        """
        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
            
            # Use proper async handling based on loop state
            if loop.is_running():
                # Create a new thread to run the async code
                import nest_asyncio
                nest_asyncio.apply()
                result = loop.run_until_complete(self.assistant_manager.query_assistant(
                    query=query,
                    thread_id=thread_id
                ))
            else:
                # Run directly if no loop is running
                result = loop.run_until_complete(self.assistant_manager.query_assistant(
                    query=query,
                    thread_id=thread_id
                ))
            
            return result["response"]
            
        except Exception as e:
            error_msg = f"Error querying OpenAI Assistant: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while searching the files: {str(e)}"
    
    async def _arun(
        self, 
        query: str,
        thread_id: Optional[str] = None,
        include_file_search: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async version of the tool execution."""
        try:
            result = await self.assistant_manager.query_assistant(
                query=query,
                thread_id=thread_id
            )
            
            return result["response"]
            
        except Exception as e:
            error_msg = f"Error querying OpenAI Assistant: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while searching the files: {str(e)}"


class FileUploadTool(BaseTool):
    """
    LangChain tool for uploading files to the OpenAI Assistant.
    """
    
    name: str = "upload_file_to_assistant"
    description: str = """Upload a file to the OpenAI Assistant for file search capabilities.
    Supported file types: PDF, TXT, DOC, DOCX, and other text-based documents.
    
    Input should be the path to the file you want to upload."""
    
    args_schema: Type[FileUploadInput] = FileUploadInput
    
    assistant_manager: Any = None
    
    def __init__(self, assistant_manager: OptimizedOpenAIAssistantManager, **kwargs):
        super().__init__(**kwargs)
        self.assistant_manager = assistant_manager
    
    @traceable(name="file_upload_tool_run")
    def _run(
        self, 
        file_path: str,
        purpose: str = "assistants",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Upload a file to the assistant.
        
        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file upload
            run_manager: Callback manager
        
        Returns:
            Success message with file information
        """
        try:
            result = self.assistant_manager.upload_file(file_path, purpose)
            
            return f"File '{result['filename']}' uploaded successfully. File ID: {result['file_id']}. The file is now available for search queries."
            
        except Exception as e:
            error_msg = f"Error uploading file: {str(e)}"
            logger.error(error_msg)
            return error_msg


# Optimized factory function for creating the tools
def create_openai_assistant_tools(assistant_id: str = None) -> List[BaseTool]:
    """
    Optimized factory function to create OpenAI Assistant tools for LangGraph integration.
    
    Args:
        assistant_id: Optional existing assistant ID to use
    
    Returns:
        List of LangChain tools for the assistant
    """
    try:
        # Try to get assistant_id from env if not provided
        if not assistant_id:
            assistant_id = get_assistant_from_env()
            if not assistant_id:
                logger.warning("No assistant ID found, will create a new one")
        
        # Create the optimized assistant tool
        assistant_tool = OptimizedOpenAIAssistantTool(assistant_id=assistant_id)
        
        # Create file upload tool with the same manager
        file_upload_tool = FileUploadTool(assistant_manager=assistant_tool.assistant_manager)
        
        logger.info("Optimized OpenAI Assistant tools created successfully")
        return [assistant_tool, file_upload_tool]
        
    except Exception as e:
        logger.error(f"Error creating OpenAI Assistant tools: {str(e)}")
        raise


# Enhanced demo function
async def demo_optimized_assistant():
    """Demo function to test the optimized OpenAI Assistant functionality."""
    print("🧪 Testing Optimized OpenAI Assistant with File Search...")
    
    try:
        # Method 1: Use existing assistant ID (recommended)
        assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
        
        if assistant_id:
            print(f"✅ Using existing assistant: {assistant_id}")
            manager = OptimizedOpenAIAssistantManager(assistant_id=assistant_id)
        else:
            print("🆕 Creating new assistant...")
            manager = OptimizedOpenAIAssistantManager(create_new_assistant=True)
        
        # Show assistant info
        info = manager.get_assistant_info()
        print(f"📋 Assistant Info: {info}")
        
        # Test query
        result = await manager.query_assistant(
            "Hello, please introduce yourself and explain your file search capabilities."
        )
        print(f"🤖 Assistant: {result['response']}")
        print(f"⏱️  Response time: {result['response_time']:.2f}s")
        print(f"🔗 Thread ID: {result['thread_id']}")
        
        # Test with thread continuity
        follow_up = await manager.query_assistant(
            "What types of files can you search through?",
            thread_id=result['thread_id']
        )
        print(f"🤖 Follow-up: {follow_up['response']}")
        
        print("✅ Optimized demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")


# Helper function to get or create assistant
def get_or_create_assistant(assistant_id: str = None, create_if_missing: bool = True) -> str:
    """
    Helper function to get existing assistant ID or create a new one.
    
    Args:
        assistant_id: Optional existing assistant ID
        create_if_missing: Whether to create a new assistant if none provided
    
    Returns:
        Assistant ID
    """
    # Try environment variable first
    if not assistant_id:
        assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
    
    # If still no ID and we're allowed to create, make a new one
    if not assistant_id and create_if_missing:
        client = OpenAI()
        assistant = client.beta.assistants.create(
            name="LangGraph Voice RAG Assistant",
            instructions="""You are an intelligent AI assistant with file search capabilities integrated into a voice-enabled RAG system.

Your capabilities include:
1. Searching through uploaded documents and files to find relevant information
2. Answering questions based on file content with high accuracy
3. Providing detailed analysis of document content
4. Citing sources from the files when providing information

Guidelines for responses:
- Always cite specific documents or sections when referencing file content
- Provide accurate and comprehensive answers based on available information
- If information isn't found in the files, clearly state that
- Keep responses concise and voice-friendly (under 100 words when possible)
- Use clear, conversational language
- Prioritize the most relevant information first

For voice interactions, be natural and conversational while maintaining accuracy.""",
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}]
        )
        assistant_id = assistant.id
        logger.info(f"Created new assistant: {assistant_id}")
        print(f"🆕 Created new assistant: {assistant_id}")
        print("💡 Save this ID to OPENAI_ASSISTANT_ID environment variable for future use")
    
    return assistant_id


def get_assistant_from_env(env_var_name: str = "OPENAI_ASSISTANT_ID") -> Optional[str]:
    """
    Get assistant ID from environment variables.
    
    Args:
        env_var_name: Name of environment variable containing assistant ID
        
    Returns:
        Assistant ID or None if not found
    """
    assistant_id = os.getenv(env_var_name)
    
    if assistant_id:
        logger.info(f"Found assistant ID in environment: {assistant_id}")
        return assistant_id
    
    # Also check for hardcoded ID if not found
    hardcoded_id = "asst_BsP2eMHSIkpyewYQQGevAsAv"  # The ID provided in user query
    logger.info(f"No assistant ID in environment, using hardcoded: {hardcoded_id}")
    return hardcoded_id


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_optimized_assistant())
    elif len(sys.argv) > 1 and sys.argv[1] == "create":
        # Create a new assistant and print the ID
        assistant_id = get_or_create_assistant(create_if_missing=True)
        print(f"Assistant ID: {assistant_id}")
    else:
        print("Optimized OpenAI Assistant module loaded.")
        print("Usage:")
        print("  python gpt_assistant.py demo    - Run demo")
        print("  python gpt_assistant.py create  - Create new assistant")
        print("")
        print("Set OPENAI_ASSISTANT_ID environment variable to use existing assistant")
