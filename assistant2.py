"""
FRIENDLY AI CHAT PROGRAM 🤖
--------------------------

What it does:
- Listens to you talk 🎤
- Answers back naturally 🗣️
- Shows text of conversation 📝
- Can be interrupted like real chat 🔄

How to use:
1. Set OPENAI_API_KEY in your computer
2. Install: pip install pyaudio websocket-client
3. Run: python thisfile.py
4. Talk normally!
5. Press Ctrl+C to exit

Made by: [Your name]
Date: [Today's date]
"""

# Import needed tools
import json         # For working with data
import base64       # For encoding audio
import pyaudio      # For recording/playing sound
import websocket    # For talking to OpenAI
import threading    # For doing many things at once
import time         # For timing things
import ssl          # For secure connection
import logging      # For tracking problems
from queue import Queue, Empty, Full  # For storing data in line
import os          # For getting API key
from dataclasses import dataclass     # For making simple classes
from typing import Optional, Dict, Any # For type hints

# Sound settings - how we capture and play audio
SAMPLE_RATE = 24000       # How many sound samples per second
CHUNK_SIZE = 1024        # How big each piece of audio is
CHANNELS = 1             # Mono sound (1) vs Stereo (2)
FORMAT = pyaudio.paInt16  # How sound is stored in computer

# OpenAI connection settings
API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
API_KEY = os.getenv('OPENAI_API_KEY')  # Get API key from computer

# Check if API key exists
if not API_KEY:
    raise EnvironmentError("❌ Please set OPENAI_API_KEY in your computer first!")

# Set up error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Colors:
    """
    Pretty colors for text in terminal 🎨
    
    Why: Makes it easy to see who's talking
    How: Add these before text, like: f"{Colors.USER}Hello{Colors.RESET}"
    """
    USER = '\033[94m'      # Blue for user
    ASSISTANT = '\033[92m'  # Green for AI
    SYSTEM = '\033[93m'     # Yellow for system messages
    ERROR = '\033[91m'      # Red for errors
    RESET = '\033[0m'       # Back to normal color

class AudioBuffer:
    """
    Stores audio data safely when multiple parts of program need it 🔒
    
    Think of it like a safe box where we can put sound and take it out,
    but only one person can use it at a time (thread-safe).
    """
    
    def __init__(self, maxsize: int = 1000):
        """
        Start new buffer
        
        Args:
            maxsize: How many pieces of audio it can hold
        """
        self.buffer = Queue(maxsize=maxsize)  # Line of audio pieces
        self.lock = threading.Lock()          # Safety lock

    def write(self, data: bytes) -> None:
        """
        Put audio in buffer 📥
        
        If full, remove oldest piece first
        """
        with self.lock:  # Lock so only one thing can write at a time
            try:
                self.buffer.put_nowait(data)
            except Full:
                # If full, remove old data and try again
                try:
                    self.buffer.get_nowait()
                    self.buffer.put_nowait(data)
                except Empty:
                    pass

    def read(self, block: bool = True, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Get audio from buffer 📤
        
        Args:
            block: Wait if empty?
            timeout: How long to wait
        
        Returns:
            Audio data or None if empty
        """
        try:
            return self.buffer.get(block=block, timeout=timeout)
        except Empty:
            return None

    def clear(self) -> None:
        """Empty the buffer 🗑️"""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except Empty:
                    break

class AudioManager:
    """
    Handles recording and playing sound 🎵
    
    This is like a DJ booth - it can record from mic and play through speakers,
    all at the same time!
    """
    
    def __init__(self) -> None:
        """Set up audio system"""
        # Main audio system
        self.audio = pyaudio.PyAudio()
        
        # Recording and playing channels
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # Places to store audio
        self.input_buffer = AudioBuffer()   # From mic
        self.output_buffer = AudioBuffer()  # To speakers
        
        # Control flags
        self.should_stop = threading.Event()      # Should we stop?
        self.user_speaking = threading.Event()    # Is user talking?
        self.assistant_speaking = threading.Event()  # Is AI talking?
        
        # Track how much audio played
        self.played_audio_bytes = 0

    def start_streams(self) -> None:
        """Start recording and playing systems"""
        # Start recording
        self.input_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._input_callback
        )

        # Start playing
        self.output_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._output_callback
        )

    def _input_callback(self, in_data: bytes, frame_count: int,
                       time_info: Dict, status: int) -> tuple:
        """
        Called when new audio comes from mic 🎤
        
        Stores it in input buffer
        """
        if not self.should_stop.is_set():
            self.input_buffer.write(in_data)
        return (None, pyaudio.paContinue)

    def _output_callback(self, in_data: bytes, frame_count: int,
                        time_info: Dict, status: int) -> tuple:
        """
        Called when speakers need more audio 🔊
        
        Sends next piece of audio to speakers
        """
        try:
            # If user is talking, play silence
            if self.user_speaking.is_set():
                return (b'\x00' * frame_count * 2, pyaudio.paContinue)

            # Get next piece of audio
            data = self.output_buffer.read(block=False)
            if data:
                self.played_audio_bytes += len(data)
                return (data, pyaudio.paContinue)
            
            # If no audio, play silence
            return (b'\x00' * frame_count * 2, pyaudio.paContinue)

        except Exception as e:
            logger.error(f"🔊 Audio output error: {e}")
            return (b'\x00' * frame_count * 2, pyaudio.paContinue)

    def clear_output(self) -> None:
        """Clear speaker buffer and reset counter"""
        self.output_buffer.clear()
        self.played_audio_bytes = 0

    def start(self) -> None:
        """Start everything"""
        self.start_streams()
        self.input_stream.start_stream()
        self.output_stream.start_stream()

    def stop(self) -> None:
        """Stop everything safely"""
        self.should_stop.set()
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        self.audio.terminate()

class ConversationManager:
    """
    Manages talking with OpenAI 🤖
    
    This is like a phone operator - handles sending/receiving messages,
    keeps track of conversation, and makes sure everything flows naturally.
    """
    
    def __init__(self, api_key: str, audio_manager: AudioManager) -> None:
        """
        Set up conversation manager
        
        Args:
            api_key: OpenAI API key
            audio_manager: System for handling audio
        """
        self.api_key = api_key
        self.audio_manager = audio_manager
        
        # Connection to OpenAI
        self.ws: Optional[websocket.WebSocketApp] = None
        
        # Control flags
        self.active = threading.Event()
        self.should_stop = threading.Event()
        self.display_lock = threading.Lock()
        
        # Connection management
        self.ws_thread: Optional[threading.Thread] = None
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30
        
        # Track current response
        self.current_response_id: Optional[str] = None
        self.current_item_id: Optional[str] = None
        
        # Store text being generated
        self.assistant_text = ""
        self.assistant_transcript = ""

    def connect(self) -> None:
        """Connect to OpenAI"""
        websocket.enableTrace(False)
        
        # Set up connection
        self.ws = websocket.WebSocketApp(
            API_URL,
            header={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start connection in background
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()

    def _configure_session(self) -> None:
        """
        Set up how AI should behave 🎭
        
        This tells AI:
        - What voice to use
        - How to detect speech
        - How to act and respond
        """
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "sage",  # Natural voice
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.4,  # More sensitive
                    "prefix_padding_ms": 200,  # Quick response
                    "silence_duration_ms": 400  # Short pause detection
                },
                "instructions": (
                    "You are a friendly, natural AI assistant. "
                    "Speak conversationally and be concise. "
                    "Stop immediately if interrupted. "
                    "Use natural voice inflections and emotions. "
                    "Remember context when interrupted."
                ),
                "temperature": 0.8  # Creativity level
            }
        }
        self.ws.send(json.dumps(config))

    def _run_websocket(self) -> None:
        """
        Keep connection alive 🔄
        
        Reconnects if connection drops
        """
        while not self.should_stop.is_set():
            try:
                self.ws.run_forever(
                    sslopt={"cert_reqs": ssl.CERT_NONE},
                    ping_interval=30,
                    ping_timeout=10
                )

                if not self.should_stop.is_set():
                    with self.display_lock:
                        print(f"\n{Colors.ERROR}Lost connection! Trying again in {self.reconnect_delay}s...{Colors.RESET}")
                    time.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

            except Exception as e:
                logger.error(f"Connection error: {e}")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    def _start_audio_processing(self) -> None:
        """
        Start sending audio to OpenAI 🎤
        
        Runs in background, constantly sending mic audio
        """
        def process_audio() -> None:
            while self.active.is_set() and not self.should_stop.is_set():
                try:
                    audio_data = self.audio_manager.input_buffer.read(timeout=0.05)
                    if audio_data and self.ws and self.ws.sock and self.ws.sock.connected:
                        event = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio_data).decode('utf-8')
                        }
                        self.ws.send(json.dumps(event))
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Audio processing error: {e}")

        threading.Thread(target=process_audio, daemon=True).start()

    def _handle_speech_started(self, event: Dict[str, Any]) -> None:
        """
        Called when user starts talking 🗣️
        
        Stops AI if it was talking
        """
        self.audio_manager.user_speaking.set()
        
        if self.audio_manager.assistant_speaking.is_set():
            # Calculate where to cut AI's response
            audio_end_ms = int((self.audio_manager.played_audio_bytes / 2) / SAMPLE_RATE * 1000)
            
            self._cancel_current_response()
            
            # Tell OpenAI where we cut off
            if self.current_item_id:
                truncate_event = {
                    "type": "conversation.item.truncate",
                    "item_id": self.current_item_id,
                    "content_index": 0,
                    "audio_end_ms": audio_end_ms
                }
                self.ws.send(json.dumps(truncate_event))

        with self.display_lock:
            print(f"\n{Colors.USER}User: [Speaking...]{Colors.RESET}")

    def _handle_speech_stopped(self, event: Dict[str, Any]) -> None:
        """Called when user stops talking"""
        self.audio_manager.user_speaking.clear()

    def _handle_response_created(self, event: Dict[str, Any]) -> None:
        """Called when AI starts new response"""
        response = event.get('response', {})
        self.current_response_id = response.get('id')
        self.audio_manager.assistant_speaking.set()
        self.audio_manager.clear_output()
        self.assistant_text = ""
        self.assistant_transcript = ""

    def _handle_response_done(self, event: Dict[str, Any]) -> None:
        """Called when AI finishes response"""
        self.audio_manager.assistant_speaking.clear()
        self.current_response_id = None
        self.current_item_id = None
        self.audio_manager.clear_output()

    def _handle_text_delta(self, event: Dict[str, Any]) -> None:
        """
        Show AI's text as it's generated 📝
        """
        delta = event.get('delta', '')
        if delta:
            self.assistant_text += delta
            with self.display_lock:
                print(f"{Colors.ASSISTANT}Assistant: {delta}{Colors.RESET}", end='', flush=True)

    def _handle_audio_delta(self, event: Dict[str, Any]) -> None:
        """
        Handle AI's voice audio 🔊
        """
        if not self.audio_manager.user_speaking.is_set():
            audio_data = base64.b64decode(event.get('delta', ''))
            self.audio_manager.output_buffer.write(audio_data)

    def _handle_audio_transcript_delta(self, event: Dict[str, Any]) -> None:
        """
        Show what AI is saying in text 📝
        """
        delta = event.get('delta', '')
        if delta:
            self.assistant_transcript += delta
            with self.display_lock:
                print(f"\n{Colors.ASSISTANT}AI saying: {self.assistant_transcript}{Colors.RESET}")

    def _handle_input_audio_transcription_completed(self, event: Dict[str, Any]) -> None:
        """
        Show what user said in text 👤
        """
        transcript = event.get('transcript', '')
        if transcript:
            with self.display_lock:
                print(f"\n{Colors.USER}You said: {transcript}{Colors.RESET}\n")

    def _handle_output_item_added(self, event: Dict[str, Any]) -> None:
        """Track AI's current response"""
        item = event.get('item', {})
        item_id = item.get('id')
        if item_id:
            self.current_item_id = item_id
        else:
            logger.warning("Missing response ID!")

    def _cancel_current_response(self) -> None:
        """
        Stop AI's current response ✋
        
        Used when interrupting AI
        """
        if self.current_response_id:
            cancel_event = {
                "type": "response.cancel"
            }
            self.ws.send(json.dumps(cancel_event))
            self.audio_manager.clear_output()
            self.audio_manager.assistant_speaking.clear()
            self.current_response_id = None
            self.current_item_id = None

    def start(self) -> None:
        """Start everything"""
        self.active.set()
        self.connect()

    def stop(self) -> None:
        """Stop everything safely"""
        self.should_stop.set()
        self.active.clear()
        if self.ws:
            self.ws.close()

    def on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle any errors"""
        logger.error(f"Connection error: {error}")
        with self.display_lock:
            print(f"\n{Colors.ERROR}Error: {error}{Colors.RESET}")

    def on_close(self, ws: websocket.WebSocketApp,
                 close_status_code: Optional[int],
                 close_msg: Optional[str]) -> None:
        """Handle connection closing"""
        self.active.clear()
        with self.display_lock:
            print(f"\n{Colors.SYSTEM}Connection closed: {close_status_code} - {close_msg}{Colors.RESET}")

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle new connection"""
        with self.display_lock:
            print(f"\n{Colors.SYSTEM}Connected to OpenAI! 🚀{Colors.RESET}")
        self._configure_session()
        self._start_audio_processing()

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """
        Handle messages from OpenAI 📨
        
        Routes each message to correct handler
        """
        try:
            event = json.loads(message)
            event_type = event.get('type', '')

            # Map of event types to their handlers
            handlers = {
                'input_audio_buffer.speech_started': self._handle_speech_started,
                'input_audio_buffer.speech_stopped': self._handle_speech_stopped,
                'response.created': self._handle_response_created,
                'response.done': self._handle_response_done,
                'response.text.delta': self._handle_text_delta,
                'response.audio.delta': self._handle_audio_delta,
                'response.audio_transcript.delta': self._handle_audio_transcript_delta,
                'conversation.item.input_audio_transcription.completed': self._handle_input_audio_transcription_completed,
                'response.output_item.added': self._handle_output_item_added
            }

            # Call correct handler or handle error
            if event_type in handlers:
                handlers[event_type](event)
            elif event_type == 'error':
                error = event.get('error', {})
                self.on_error(ws, Exception(error.get('message', 'Unknown error')))
            else:
                logger.debug(f"Skipped event: {event_type}")

        except Exception as e:
            logger.error(f"Message error: {e}")
            with self.display_lock:
                print(f"\n{Colors.ERROR}Error: {e}{Colors.RESET}")

def main() -> None:
    """
    Main program 🎯
    
    Starts everything and handles clean shutdown
    """
    try:
        # Start audio system
        audio_manager = AudioManager()
        audio_manager.start()

        # Start conversation system
        conversation_manager = ConversationManager(API_KEY, audio_manager)
        conversation_manager.start()

        # Show welcome message
        print(f"\n{Colors.SYSTEM}AI ready! 🤖{Colors.RESET}")
        print(f"{Colors.SYSTEM}Start talking...{Colors.RESET}")
        print(f"{Colors.SYSTEM}Press Ctrl+C to quit{Colors.RESET}")

        # Keep running until Ctrl+C
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n{Colors.SYSTEM}Shutting down...{Colors.RESET}")
    except Exception as e:
        logger.error(f"Big error: {e}")
        print(f"\n{Colors.ERROR}Big error: {e}{Colors.RESET}")
    finally:
        # Clean up
        if 'conversation_manager' in locals():
            conversation_manager.stop()
        if 'audio_manager' in locals():
            audio_manager.stop()
        print(f"{Colors.SYSTEM}Bye! 👋{Colors.RESET}")

if __name__ == "__main__":
    main()