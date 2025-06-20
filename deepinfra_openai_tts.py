import os
import tempfile
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Optional, AsyncIterator, Dict
from openai import OpenAI
from livekit.agents import tts
from livekit.agents.tts import TTSCapabilities, AudioEmitter
from livekit.agents import APIConnectOptions
from livekit import rtc
import pygame

# Define default connection options with reduced timeout
DEFAULT_CONN_OPTIONS = APIConnectOptions(
    timeout=15.0,  # Reduced from 30.0
)

# TTS Cache to avoid regenerating the same audio
TTS_CACHE: Dict[str, Path] = {}
# Hard limit on TTS input length for faster processing
MAX_TTS_LENGTH = 100

class DeepInfraOpenAITTS(tts.TTS):
    """An optimized TTS implementation using caching and input limiting"""
    
    def __init__(
        self,
        *,
        voice: str = "tara",
        model: str = "canopylabs/orpheus-3b-0.1-ft",  # Using smallest model for faster processing
        api_key: Optional[str] = None,
        sample_rate: int = 24000
    ):
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        
        self._voice = voice
        self._model = model
        self._api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "DeepInfra API key is required. Set DEEPINFRA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize OpenAI client for DeepInfra with shorter timeout
        self._client = OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=self._api_key,
            timeout=10.0  # Add global timeout
        )
        
        # Set up temp directory for audio files with cache
        self._temp_dir = Path(tempfile.gettempdir()) / "deepinfra_tts"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Init pygame mixer for audio playback
        pygame.mixer.init()
        
        # Clear older cache files to prevent disk filling
        self._clear_old_cache_files()
        
    def _clear_old_cache_files(self):
        """Remove TTS cache files older than 1 day"""
        try:
            current_time = time.time()
            for file_path in self._temp_dir.glob("speech_*.mp3"):
                if current_time - file_path.stat().st_mtime > 86400:  # 24 hours
                    try:
                        os.remove(file_path)
                    except:
                        pass
        except:
            # Don't let cleanup errors block execution
            pass

    def synthesize(
        self, 
        text: str, 
        *, 
        conn_options: Optional[APIConnectOptions] = None
    ) -> tts.ChunkedStream:
        """Synthesize text to speech with length limit for performance"""
        # Truncate long text to improve performance
        if len(text) > MAX_TTS_LENGTH:
            # Find a good breakpoint (sentence or paragraph end)
            breakpoint = MAX_TTS_LENGTH
            for end_char in ['.', '!', '?', '\n']:
                last_end = text[:MAX_TTS_LENGTH].rfind(end_char)
                if last_end > 0:
                    breakpoint = last_end + 1
                    break
            text = text[:breakpoint]
        
        # Use default connection options if none provided
        if conn_options is None:
            conn_options = DEFAULT_CONN_OPTIONS
            
        return OptimizedTTSStream(
            tts=self,
            input_text=text,
            client=self._client,
            voice=self._voice,
            model=self._model,
            temp_dir=self._temp_dir,
            conn_options=conn_options
        )

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def model(self) -> str:
        return self._model


class OptimizedTTSStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: DeepInfraOpenAITTS,
        input_text: str,
        client: OpenAI,
        voice: str,
        model: str,
        temp_dir: Path,
        conn_options: APIConnectOptions,
    ):
        # Pass conn_options to parent class
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._voice = voice
        self._model = model
        self._temp_dir = temp_dir
        
        # Create a hash of the input text for caching
        self._text_hash = hashlib.md5(input_text.encode()).hexdigest()
        
    async def _run(self, output_emitter: AudioEmitter) -> None:
        request_id = f"deepinfra-{id(self)}"
        
        # Initialize the emitter
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/mp3",
        )
        
        try:
            # Check cache first to avoid regenerating same audio
            speech_file = self._temp_dir / f"speech_{self._text_hash}.mp3"
            cache_key = f"{self._voice}_{self._text_hash}"
            
            # If audio already exists in cache, use it directly
            if cache_key in TTS_CACHE and TTS_CACHE[cache_key].exists():
                speech_file = TTS_CACHE[cache_key]
            elif speech_file.exists():
                # File exists but not in memory cache
                TTS_CACHE[cache_key] = speech_file
            else:
                # Not in cache, generate with shorter timeout
                timeout = 10.0  # Shorter timeout for faster failure/retry
                
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
                    
                    # Add to cache
                    TTS_CACHE[cache_key] = speech_file
                
                # Run with shorter timeout
                await asyncio.wait_for(generate_speech(), timeout=timeout)
            
            # Read the audio file (either from cache or newly generated)
            with open(speech_file, "rb") as f:
                audio_data = f.read()
            
            # Push the audio data to the emitter
            output_emitter.push(audio_data)
            
        except asyncio.TimeoutError:
            # Use simple fallback to avoid delay
            await self._fast_fallback(output_emitter)
        except Exception as e:
            # Use fallback on any error
            await self._fast_fallback(output_emitter)
    
    async def _fast_fallback(self, output_emitter):
        """Immediate fallback without trying complex recovery"""
        try:
            # Use a pre-generated fallback file if possible
            fallback_file = self._temp_dir / "fallback_alert.mp3"
            
            if not fallback_file.exists():
                # Generate a very short message
                short_text = "I'll respond in text."
                await self._client.audio.speech.create(
                    model=self._model,
                    voice=self._voice,
                    input=short_text,
                    response_format="mp3",
                ).stream_to_file(fallback_file)
            
            # Push the fallback audio
            with open(fallback_file, "rb") as f:
                audio_data = f.read()
            output_emitter.push(audio_data)
        except:
            # If even this fails, just continue silently
            pass