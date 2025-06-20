## 📁 Project Structure

### Core Files

#### `voice_agent.py`
**Enhanced LiveKit Voice Agent with Production-Grade Reliability**

This is the main voice interaction component featuring:
- **VoiceAssistant Class**: Complete voice interaction management
- **ReliableDeepgramSTT**: Enhanced STT with retry logic and error recovery
- **ReliableTTS**: Custom TTS implementation with fallback mechanisms
- **Error Monitoring**: Automatic error detection and recovery systems
- **Session Management**: Robust session initialization and maintenance
- **Parallel TTS**: Optimized for faster response times

Key features:
- Silero VAD with optimized settings for real-time performance
- Multi-retry STT/TTS with exponential backoff
- Automatic API key validation
- Connection health monitoring
- Graceful error recovery

#### `gpt_assistant.py`
**OpenAI Assistant Integration with LangChain Tools**

Provides document search and RAG capabilities:
- **OptimizedOpenAIAssistantManager**: Efficient assistant management
- **LangChain Tool Integration**: Seamless tool wrapping for workflows  
- **File Upload Management**: Document processing and indexing
- **Thread Continuity**: Persistent conversation contexts
- **Enhanced Error Handling**: Robust API interaction patterns

Features:
- File search across multiple document types (PDF, TXT, DOC, etc.)
- Streaming responses for better user experience
- Conversation memory and context management
- Traceable operations with LangSmith integration
- Optimized for voice-friendly responses

#### `deepinfra_openai_tts.py`
**High-Performance TTS with Caching and Optimization**

Custom TTS implementation featuring:
- **DeepInfraOpenAITTS**: Main TTS class with caching
- **OptimizedTTSStream**: Streaming audio with retry logic
- **Response Caching**: Hash-based audio caching system
- **Performance Optimization**: Reduced timeouts and parallel processing
- **Fallback Mechanisms**: Multiple recovery strategies

Optimizations:
- Input length limiting for faster processing
- Disk-based caching with automatic cleanup
- Shortened timeouts for quicker failures/retries
- Memory-efficient streaming
- Pygame integration for audio playback

#### `main_demo_langgraph.py`
**LangGraph-Powered Voice Agent with React Pattern**

Advanced AI workflow orchestration:
- **ReactAgentState**: Enhanced state management for complex workflows
- **LangGraphVoiceAgent**: Core reasoning and tool orchestration
- **LangGraphVoiceAssistant**: Complete voice assistant implementation  
- **Tool Integration**: Seamless OpenAI Assistant integration
- **Memory Management**: Persistent conversation history

Features:
- React agent pattern for advanced reasoning
- State management with LangGraph
- Tool execution with error handling
- Voice-optimized responses
- Thread management for conversation continuity

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- LiveKit Cloud account or self-hosted server
- API keys for:
  - OpenAI (for Assistant and GPT models)
  - Google AI (for Gemini models)
  - Deepgram (for STT)
  - DeepInfra (for TTS)
  - Tavily (for web search - optional)

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd livekit-demo
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```
4. **Live-kit plugins**:
```
pip install \
  "livekit-agents[deepgram,openai,cartesia,silero,turn-detector]~=1.0" \
  "livekit-plugins-noise-cancellation~=0.2" \
  "python-dotenv"
```

Required environment variables:
```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_ASSISTANT_ID=your_assistant_id  # Optional, will create if not provided

# Google AI
GOOGLE_API_KEY=your_google_api_key

# Deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key

# DeepInfra
DEEPINFRA_API_KEY=your_deepinfra_api_key

# LiveKit (if using LiveKit Cloud)
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_secret

# Optional
TAVILY_API_KEY=your_tavily_api_key  # For web search
LANGSMITH_API_KEY=your_langsmith_key  # For tracing
```

4. **Download model weights** (for Silero VAD):
```bash
python voice_agent.py download-files
```

### Running the Assistant

#### Option 1: Basic Voice Agent
```bash
python voice_agent.py
```

#### Option 2: LangGraph-powered Agent
```bash
python main_demo_langgraph.py
```

#### Option 3: OpenAI Assistant Demo
```bash
python gpt_assistant.py demo
```

## 📖 Usage Examples

### Basic Voice Interaction
```python
from voice_agent import VoiceAssistant
import asyncio

async def main():
    assistant = VoiceAssistant(
        instructions="You are a helpful voice assistant.",
        initial_greeting="Hello! I'm your AI assistant. How can I help you today?"
    )
    
    # This would be called by LiveKit
    # await assistant.run(ctx)

if __name__ == "__main__":
    asyncio.run(main())
```

### Document Search with OpenAI Assistant
```python
from gpt_assistant import OptimizedOpenAIAssistantManager
import asyncio

async def search_documents():
    manager = OptimizedOpenAIAssistantManager(
        assistant_id="your_assistant_id",
        create_new_assistant=True
    )
    
    # Upload a document
    file_info = manager.upload_file("path/to/document.pdf")
    print(f"Uploaded: {file_info['filename']}")
    
    # Search the document
    result = await manager.query_assistant(
        "What are the key points in this document?"
    )
    print(f"Response: {result['response']}")

if __name__ == "__main__":
    asyncio.run(search_documents())
```

### LangGraph Workflow
```python
from main_demo_langgraph import LangGraphVoiceAgent
import asyncio

async def use_langgraph_agent():
    agent = LangGraphVoiceAgent(
        model_name="gpt-4o-mini",
        temperature=0.7
    )
    
    response = await agent.process_voice_query(
        "Search for information about AI safety",
        thread_id="user_session_1"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(use_langgraph_agent())
```

## 🔧 Configuration

### Voice Agent Settings
```python
assistant = VoiceAssistant(
    instructions="Custom instructions for your assistant",
    google_model="gemini-2.5-flash-lite-preview-06-17",
    deepinfra_model="canopylabs/orpheus-3b-0.1-ft",
    deepinfra_voice="tara",
    deepgram_model="nova-2",
    enable_parallel_tts=True  # For faster responses
)
```

### Silero VAD Optimization
The project includes optimized VAD settings to prevent "inference slower than realtime" warnings:
```python
vad=silero.VAD.load(
    force_cpu=True,
    activation_threshold=0.6,
    min_silence_duration=0.5,
    sample_rate=8000,  # Lower sample rate for faster processing
)
```

### TTS Performance Tuning
```python
tts = ReliableTTS(
    voice="tara",
    model="canopylabs/orpheus-3b-0.1-ft",
    max_retries=3,
    retry_delay=0.1
)
```

## 🛠️ Development

### Adding New Tools
```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel

class CustomToolInput(BaseModel):
    query: str

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Description of what the tool does"
    args_schema = CustomToolInput
    
    def _run(self, query: str) -> str:
        # Tool implementation
        return f"Result for: {query}"
```

### Custom LLM Integration
```python
from langchain_community.llms import YourCustomLLM

def get_custom_llm():
    return YourCustomLLM(
        api_key="your_api_key",
        model="your_model",
        temperature=0.7
    )
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Voice Agent Testing
```bash
python voice_agent.py --test-mode
```

### OpenAI Assistant Testing
```bash
python gpt_assistant.py demo
```

## 📊 Performance Optimization

### TTS Optimization
- **Caching**: Responses are cached to avoid regeneration
- **Input Limiting**: Long text is truncated for faster processing
- **Parallel Processing**: Multiple TTS requests can be handled concurrently
- **Timeout Management**: Aggressive timeouts with fallback mechanisms

### STT Optimization
- **Retry Logic**: Exponential backoff for failed requests
- **Connection Pooling**: Efficient connection management
- **Error Recovery**: Automatic reconnection on failures

### VAD Optimization
- **CPU Processing**: Force CPU usage for consistent performance
- **Sample Rate**: Optimized 8kHz sample rate for speed
- **Threshold Tuning**: Balanced sensitivity for accurate detection

## 🚨 Troubleshooting

### Common Issues

#### "Inference slower than realtime" Warning
This has been fixed with optimized VAD settings:
```python
vad=silero.VAD.load(
    force_cpu=True,
    sample_rate=8000,
    activation_threshold=0.6
)
```

#### TTS Timeout Errors
The project includes comprehensive retry logic and fallback mechanisms:
- Multiple retry attempts with exponential backoff
- Fallback TTS with shorter messages
- Graceful degradation to text-only responses

#### STT Connection Issues
Enhanced error handling includes:
- Automatic connection recovery
- Multiple retry attempts
- Alternative STT provider fallback

#### Memory Issues
- Automatic cache cleanup for TTS files
- Efficient memory management in streaming
- Garbage collection for long-running sessions

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

- **API Key Management**: Store keys in environment variables
- **Input Validation**: All user inputs are validated and sanitized
- **Rate Limiting**: Built-in rate limiting for API calls
- **Error Handling**: Errors don't expose sensitive information
- **File Upload Security**: Validate file types and sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include comprehensive error handling
- Write unit tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LiveKit**: For excellent real-time communication infrastructure
- **LangChain/LangGraph**: For powerful AI workflow orchestration
- **OpenAI**: For advanced language models and assistant capabilities
- **Deepgram**: For high-quality speech-to-text services
- **DeepInfra**: For cost-effective AI model hosting

## 📚 Additional Resources

- [LiveKit Documentation](https://docs.livekit.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI Assistant API](https://platform.openai.com/docs/assistants/overview)
- [Deepgram API Documentation](https://developers.deepgram.com/)

## 🆕 Recent Updates

### v1.2.0
- Fixed Silero VAD performance warnings
- Enhanced error handling across all components
- Added comprehensive retry mechanisms
- Improved TTS caching system
- Optimized for production deployment

### v1.1.0
- Added LangGraph integration
- Implemented OpenAI Assistant RAG capabilities
- Enhanced voice optimization
- Added comprehensive logging

### v1.0.0
- Initial release with basic voice agent functionality
- LiveKit integration
- Multi-provider TTS/STT support

---

**Built with ❤️ for the AI community**

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/your-username/livekit-demo).
