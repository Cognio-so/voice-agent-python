from dotenv import load_dotenv
import os
import asyncio
import logging
import time
from typing import Optional, List, Callable, Any

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
from livekit.agents import stt, tts
from livekit.plugins import (
    google,
    openai,
    noise_cancellation,
    silero,
    speechify,
    deepgram,
)

# Import our custom TTS implementation
from deepinfra_openai_tts import DeepInfraOpenAITTS

# Set up logging to see more detailed error messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_assistant")

load_dotenv()


@function_tool
async def lookup_information(
    context: RunContext,
    query: str,
):
    """Used to look up information about a topic."""
    return f"Here's some information about {query}: This is placeholder information. In a real implementation, this would connect to a search API."


class ReliableDeepgramSTT(deepgram.STT):
    """Enhanced Deepgram STT implementation with better error handling"""
    
    def __init__(
        self,
        model: str = "nova-2", 
        api_key: Optional[str] = None,
        language: str = "multi",
        max_retries: int = 1,
        retry_delay: float = 0.5
    ):
        super().__init__(model=model, api_key=api_key, language=language)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    async def _run(self):
        """Override _run method to add retry logic"""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                return await super()._run()
            except Exception as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    logger.warning(f"STT error (attempt {retries}/{self.max_retries}): {str(e)}. Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    # Increase retry delay with each attempt (exponential backoff)
                    self.retry_delay *= 0.05
        
        logger.error(f"STT failed after {self.max_retries} attempts: {str(last_error)}")
        # Try to restart deepgram connection
        try:
            logger.info("Attempting to restart Deepgram connection...")
            # Implementation may need to be adjusted based on internal Deepgram implementation
            return await super()._run()
        except Exception as e:
            logger.error(f"Failed to restart Deepgram connection: {str(e)}")
            raise


class ReliableTTS(DeepInfraOpenAITTS):
    """Enhanced TTS implementation with better error handling and fallback"""
    
    def __init__(
        self,
        *,
        voice: str = "tara",
        model: str = "canopylabs/orpheus-3b-0.1-ft",
        api_key: Optional[str] = None,
        sample_rate: int = 24000,
        max_retries: int = 1,
        retry_delay: float = 0.2
    ):
        super().__init__(
            voice=voice,
            model=model,
            api_key=api_key,
            sample_rate=sample_rate
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def synthesize(
        self, 
        text: str, 
        *, 
        conn_options: Optional[agents.APIConnectOptions] = None
    ) -> tts.ChunkedStream:
        """Override synthesize to add resilience"""
        # Use extended timeout for connections
        if conn_options is None:
            conn_options = agents.APIConnectOptions(timeout=60.0)
            
        logger.info(f"Synthesizing speech: '{text[:5]}...' with {self.max_retries} max retries")
        return ReliableTTSStream(
            tts=self,
            input_text=text,
            client=self._client,
            voice=self._voice,
            model=self._model,
            temp_dir=self._temp_dir,
            conn_options=conn_options,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )


class ReliableTTSStream(tts.ChunkedStream):
    """Enhanced TTS Stream with retry capabilities"""
    
    def __init__(
        self,
        *,
        tts: ReliableTTS,
        input_text: str,
        client,
        voice: str,
        model: str,
        temp_dir,
        conn_options,
        max_retries: int = 1,
        retry_delay: float = 0.2
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._voice = voice
        self._model = model
        self._temp_dir = temp_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Enhanced run with retry logic"""
        request_id = f"deepinfra-{id(self)}"
        
        # Initialize the emitter
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/mp3",
        )
        
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Generate a unique filename
                speech_file = self._temp_dir / f"speech_{hash(self._input_text)}_{retries}.mp3"
                
                # Get timeout from connection options
                timeout = self._conn_options.timeout if hasattr(self, '_conn_options') else 30.0
                
                # Create a task with timeout
                async def generate_speech():
                    loop = asyncio.get_event_loop()
                    
                    # Run the OpenAI client call in a thread to avoid blocking
                    await loop.run_in_executor(
                        None,
                        lambda: self._client.audio.speech.create(
                            model=self._model,
                            voice=self._voice,
                            input=self._input_text,
                            response_format="mp3",
                        ).stream_to_file(speech_file)
                    )
                    
                    # Read the generated audio file
                    with open(speech_file, "rb") as f:
                        audio_data = f.read()
                    
                    # Push the audio data to the emitter
                    output_emitter.push(audio_data)
                
                # Run with timeout
                await asyncio.wait_for(generate_speech(), timeout=timeout)
                break  # If successful, break the retry loop
                
            except (asyncio.TimeoutError, Exception) as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    logger.warning(f"TTS error (attempt {retries}/{self.max_retries}): {str(e)}. Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    # Exponential backoff
                    self.retry_delay *= 0.05
        
        if retries >= self.max_retries:
            # If still failing after all retries, use fallback mechanism
            logger.error(f"TTS failed after {self.max_retries} attempts: {str(last_error)}")
            try:
                # Attempt to use a simpler/backup TTS method
                await self._fallback_tts(output_emitter)
            except Exception as e:
                raise tts.TTSError(f"TTS completely failed - both primary and fallback methods: {str(e)}")
    
    async def _fallback_tts(self, output_emitter: tts.AudioEmitter) -> None:
        """Fallback TTS implementation when main one fails"""
        logger.info("Using fallback TTS mechanism")
        # Simplified message to convey the essence
        fallback_text = "I'm having trouble generating audio right now. Let me try again or you can type your next query."
        
        try:
            # Use a more reliable but lower quality approach
            speech_file = self._temp_dir / "fallback_speech.mp3"
            
            # Simple solution: Use a pre-generated audio file if available
            if not os.path.exists(speech_file):
                # Create a basic TTS using whatever method is most reliable
                # Here simulating with original method but with shorter text
                await asyncio.wait_for(
                    self._client.audio.speech.create(
                        model=self._model,
                        voice=self._voice,
                        input=fallback_text,
                        response_format="mp3",
                    ).stream_to_file(speech_file),
                    timeout=10.0  # Short timeout
                )
            
            # Push the fallback audio
            with open(speech_file, "rb") as f:
                audio_data = f.read()
            output_emitter.push(audio_data)
            
        except Exception:
            # In worst case, just log the failure but don't crash the application
            logger.error("Even fallback TTS failed - continuing without audio response")
            # We intentionally don't re-raise here, allowing the assistant to continue functioning


class VoiceAssistant:
    def __init__(
        self,
        instructions: str = """You are a helpful voice AI assistant. 
            You can answer questions, provide information, and engage in conversation.
            Be concise, friendly, and helpful in your responses.""",
        tools: List[Callable] = None,
        google_model: str = "gemini-2.5-flash-lite-preview-06-17",
        deepinfra_model: str = "canopylabs/orpheus-3b-0.1-ft",
        deepinfra_voice: str = "tara",
        deepgram_model: str = "nova-2",
        initial_greeting: str = "Greet the user warmly, introduce yourself as a voice assistant, and offer your assistance.",
        enable_parallel_tts: bool = False
    ):
        self.instructions = instructions
        self.tools = tools or [lookup_information]
        self.google_model = google_model
        self.deepinfra_model = deepinfra_model
        self.deepinfra_voice = deepinfra_voice
        self.deepgram_model = deepgram_model
        self.initial_greeting = initial_greeting
        self.enable_parallel_tts = enable_parallel_tts
        
        self.session = None
        self.agent_instance = None
        self.stt_error_count = 0  # Track STT errors
        self.tts_error_count = 0  # Track TTS errors
        self.tts_semaphore = asyncio.Semaphore(1)  # For handling concurrent TTS requests
        
    def _create_agent(self) -> Agent:
        """Create the agent instance with instructions and tools"""
        # Create an agent with error handling for responses
        agent = Agent(
            instructions=f"""
            {self.instructions}
            
            If you notice that I'm having trouble with audio input or output, suggest that I try text input 
            or mention that there might be temporary connection issues. 
            
            If I ask about problems with audio, explain that occasional network or API issues might 
            occur and the system will try to recover automatically.
            """,
            tools=self.tools
        )
        return agent
        
    async def _verify_api_keys(self) -> bool:
        """Verify that all required API keys are available"""
        required_keys = {
            "DEEPINFRA_API_KEY": os.getenv("DEEPINFRA_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY")
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            for key in missing_keys:
                logger.error(f"{key} not found in environment variables")
            logger.error("Please add missing keys to your .env file and try again.")
            return False
            
        logger.info("All API keys found. Initializing agent...")
        return True
        
    async def initialize_session(self) -> Optional[AgentSession]:
        """Initialize the agent session with all required components and enhanced error handling"""
        if not await self._verify_api_keys():
            return None
            
        # Get API keys
        deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        
        # Create the agent session with enhanced components
        self.session = AgentSession(
            # Voice activity detection
            vad=silero.VAD.load(
                force_cpu=True,
                activation_threshold=0.5, 
                min_silence_duration=0.5,  
                sample_rate=16000, 
            ),
            
            # Enhanced STT with retry logic
            stt=ReliableDeepgramSTT(
                model=self.deepgram_model, 
                api_key=deepgram_api_key,
                language="multi",
                max_retries=1,
                retry_delay=0.05
            ),
            
            # Language model with increased timeout
            llm=google.LLM(
                model=self.google_model, 
                api_key=google_api_key
            ),
            
            # Enhanced TTS with retry logic
            tts=ReliableTTS(
                voice=self.deepinfra_voice,  
                model=self.deepinfra_model, 
                api_key=deepinfra_api_key,
                max_retries=1,
                retry_delay=0.05
            )
        )
        
        return self.session
        
    async def _setup_parallel_tts_for_web_search(self):
        """Set up parallel TTS functionality with better timing."""
        if not self.enable_parallel_tts or not self.session:
            return
        
        for tool in self.tools:
            if hasattr(tool, "__name__") and tool.__name__ == "web_search":
                original_web_search = tool
                
                @function_tool
                async def enhanced_web_search(context: RunContext, query: str):
                    """Enhanced web search with faster parallel TTS"""
                    is_first_chunk = True
                    tts_task = None
                    
                    # Define the streaming callback
                    def streaming_callback(partial_text):
                        nonlocal is_first_chunk, tts_task
                        
                        # Process any meaningful chunk immediately
                        if is_first_chunk and len(partial_text) > 15:
                            is_first_chunk = False
                            
                            # Launch TTS immediately with minimal content
                            if self.session:
                                try:
                                    # Start TTS within 50ms with first sentence
                                    first_sentence = partial_text.split('.')[0] + '.'
                                    
                                    async def process_tts():
                                        try:
                                            await self.session.generate_reply(
                                                text=first_sentence, 
                                                interrupt=True
                                            )
                                        except Exception as e:
                                            logger.error(f"Parallel TTS error: {e}")
                                    
                                    tts_task = asyncio.create_task(process_tts())
                                except Exception as e:
                                    logger.error(f"Error scheduling TTS: {e}")
                    
                    # Call original search with minimal timeout (3s)
                    try:
                        result = await asyncio.wait_for(
                            original_web_search(context, query),
                            timeout=1.0
                        )
                        return result
                    except asyncio.TimeoutError:
                        # Return partial result while search continues in background
                        return "I'm looking into that now. Here's what I know so far..."

    async def start(self, ctx: agents.JobContext) -> None:
        """Start the agent in the provided room context with error handling"""
        # Initialize session if not already done, with retry
        retry_count = 0
        while not self.session and retry_count < 2:
            if await self.initialize_session():
                break
            retry_count += 1
            if retry_count < 3:
                logger.warning(f"Session initialization failed. Retrying ({retry_count}/3)...")
                await asyncio.sleep(1)
        
        if not self.session:
            logger.error("Failed to initialize session after 3 attempts.")
            return
        
        # Set up parallel TTS for web search if enabled
        if self.enable_parallel_tts:
            await self._setup_parallel_tts_for_web_search()
                
        # Create agent instance
        self.agent_instance = self._create_agent()
        
        try:
            # Start the session
            await self.session.start(
                room=ctx.room,
                agent=self.agent_instance,
                room_input_options=RoomInputOptions(
                    # Noise cancellation
                    noise_cancellation=noise_cancellation.BVC(),
                ),
            )
            
            # Connect to the room
            await ctx.connect()
            
            # Generate initial greeting with retry
            await self._generate_greeting_with_retry()
            
        except Exception as e:
            logger.error(f"Error starting assistant: {str(e)}")
            # Try to recover by reconnecting
            try:
                logger.info("Attempting to reconnect...")
                await ctx.connect()
            except Exception as reconnect_error:
                logger.error(f"Reconnect failed: {str(reconnect_error)}")
    
    async def _generate_greeting_with_retry(self, max_attempts=1):
        """Generate initial greeting with retry mechanism"""
        for attempt in range(max_attempts):
            try:
                await self.session.generate_reply(
                    instructions=self.initial_greeting
                )
                logger.info("✅ Initial greeting generated successfully!")
                return
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Error generating initial greeting (attempt {attempt+1}/{max_attempts}): {str(e)}")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Failed to generate initial greeting after {max_attempts} attempts: {str(e)}")
    
    async def _monitor_for_errors(self):
        """Monitor for repeated errors and attempt recovery"""
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            # If multiple STT or TTS errors accumulate, try to recover
            if self.stt_error_count > 1 or self.tts_error_count > 1:
                logger.warning(f"Detected multiple errors (STT: {self.stt_error_count}, TTS: {self.tts_error_count}). Attempting recovery...")
                
                try:
                    # Reset error counters
                    self.stt_error_count = 0
                    self.tts_error_count = 0
                    
                    # Attempt to notify user of the issue
                    if self.session:
                        try:
                            await self.session.generate_reply(
                                instructions="Inform the user that there may be temporary connection issues, and that you're working to resolve them."
                            )
                        except Exception:
                            pass  # Ignore errors in the notification itself
                    
                except Exception as e:
                    logger.error(f"Error during recovery attempt: {str(e)}")
            
    async def run(self, ctx: agents.JobContext) -> None:
        """Run the agent and keep it alive until terminated with enhanced error recovery"""
        await self.start(ctx)
        
        if not self.session or not self.agent_instance:
            logger.error("Failed to start the agent properly.")
            return
        
        # Start the error monitoring task
        error_monitoring_task = asyncio.create_task(self._monitor_for_errors())
        
        # Keep the session running until terminated
        try:
            logger.info("🎤 Agent is now listening. Press Ctrl+B to toggle between Text/Audio mode.")
            
            # Use a simple loop to keep the session alive with health checks
            while True:
                try:
                    await asyncio.sleep(1)
                    
                except Exception as loop_error:
                    logger.error(f"Loop error: {str(loop_error)}")
                    # Try to continue despite the error
        
        except KeyboardInterrupt:
            logger.info("\n👋 Session terminated by user. Goodbye!")
        
        except Exception as e:
            logger.error(f"\n❌ Error in main loop: {str(e)}")
        
        finally:
            # Clean up
            error_monitoring_task.cancel()
            try:
                await error_monitoring_task
            except asyncio.CancelledError:
                pass


# Updated entrypoint function
async def entrypoint(ctx: agents.JobContext):
    """Enhanced entrypoint with improved error handling"""
    try:
        assistant = VoiceAssistant()
        await assistant.run(ctx)
    except Exception as e:
        logger.critical(f"Critical error in entrypoint: {str(e)}")


if __name__ == "__main__":
    # Run the app with improved logging
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
