import os
import asyncio
import torch
import ffmpeg
import yt_dlp
import whisper
from pydub import AudioSegment
from deep_translator import GoogleTranslator
from edge_tts import Communicate
from typing import List, Tuple, Optional
import time
import logging

# ======================
# 1. CONFIGURATION
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Supported Languages (with Tamil added)
EDGE_VOICES = {
    "en": ["en-US-AriaNeural", "en-US-GuyNeural"],
    "ta": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"],  # Tamil voices
    "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "es": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"]
}

# ======================
# 2. MAIN DUBBING CLASS
# ======================
class LiveTamilDubbingTool:
    def __init__(self):
        self.cleanup_files = []
        self.live_active = False
        self.segment_duration = 15  # Seconds per segment (reduced latency)
        self.min_silence = 0.1  # Reduced silence gap for faster output
        
        # Initialize Whisper (small for speed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = whisper.load_model("small", device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Output directory
        self.output_dir = "tamil_dubbed_output"
        os.makedirs(self.output_dir, exist_ok=True)

    # ======================
    # 3. CORE FUNCTIONS
    # ======================
    async def download_segment(self, url: str) -> str:
        """Download live stream segment with low latency"""
        output_path = os.path.join(self.output_dir, f"segment_{int(time.time())}.%(ext)s")
        
        ydl_opts = {
            "format": "best[height<=480]/best",  # Lower resolution = faster download
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 10,  # Faster timeout
            "retries": 2,
            "fragment_retries": 2,
            "buffersize": 512 * 1024,  # 512KB buffer for lower latency
            "http_chunk_size": 1024 * 1024,  # 1MB chunks
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info.get('is_live', False):
                    raise ValueError("URL is not a live stream")
                
                # Download the segment
                ydl.download([url])
            
            # Find the actual downloaded file
            downloaded_file = None
            for file in os.listdir(self.output_dir):
                if file.startswith(f"segment_{int(time.time()) - 5}"):  # Allow 5 second window
                    downloaded_file = os.path.join(self.output_dir, file)
                    break
            
            if not downloaded_file or not os.path.exists(downloaded_file):
                # Fallback: find most recent segment file
                segment_files = [f for f in os.listdir(self.output_dir) if f.startswith("segment_")]
                if segment_files:
                    segment_files.sort(key=lambda x: os.path.getctime(os.path.join(self.output_dir, x)), reverse=True)
                    downloaded_file = os.path.join(self.output_dir, segment_files[0])
                else:
                    raise FileNotFoundError("Download failed - no file found")
            
            self.cleanup_files.append(downloaded_file)
            logger.info(f"Downloaded segment: {downloaded_file}")
            return downloaded_file
            
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            raise

    async def extract_audio(self, video_path: str) -> str:
        """Fast audio extraction using FFmpeg"""
        audio_path = os.path.join(self.output_dir, f"audio_{int(time.time())}.wav")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec="pcm_s16le", ar=16000, ac=1)
                .global_args("-loglevel", "error")
                .overwrite_output()
                .run(cmd=["ffmpeg", "-nostdin"])
            )
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio extraction failed")
            
            self.cleanup_files.append(audio_path)
            logger.info(f"Extracted audio: {audio_path}")
            return audio_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e}")
            raise
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            raise

    async def transcribe(self, audio_path: str) -> Tuple[List[str], List[Tuple[float, float]], str]:
        """Optimized Whisper transcription"""
        try:
            def _transcribe():
                return self.model.transcribe(
                    audio_path,
                    fp16=torch.cuda.is_available(),
                    temperature=0.0,  # Disable sampling for consistency
                    compression_ratio_threshold=2.4,  # Faster processing
                    no_speech_threshold=0.6,
                    word_timestamps=False  # Disable for speed
                )
            
            result = await asyncio.get_event_loop().run_in_executor(None, _transcribe)
            
            if not result.get("segments"):
                logger.warning("No speech detected in audio")
                return [], [], "en"
            
            transcript = [seg["text"].strip() for seg in result["segments"] if seg["text"].strip()]
            timestamps = [(seg["start"], seg["end"]) for seg in result["segments"] if seg["text"].strip()]
            language = result.get("language", "en")
            
            logger.info(f"Transcribed {len(transcript)} segments in language: {language}")
            return transcript, timestamps, language
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    async def translate_to_tamil(self, texts: List[str], source_lang: str) -> List[str]:
        """Google Translate to Tamil (ta)"""
        if not texts:
            return []
        
        try:
            translator = GoogleTranslator(source=source_lang, target="ta")
            translated = []
            
            for text in texts:
                if text.strip():
                    try:
                        translated_text = translator.translate(text)
                        translated.append(translated_text)
                        logger.debug(f"Translated: '{text}' -> '{translated_text}'")
                    except Exception as e:
                        logger.warning(f"Translation failed for '{text}': {e}")
                        translated.append(text)  # Fallback to original
                else:
                    translated.append("")
            
            logger.info(f"Translated {len(translated)} segments to Tamil")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise

    async def generate_tamil_tts(self, texts: List[str]) -> List[AudioSegment]:
        """Generate Tamil TTS in parallel"""
        async def _generate(text: str, idx: int):
            if not text.strip():
                return None, None
                
            tts_file = os.path.join(self.output_dir, f"tts_{idx}_{int(time.time())}.mp3")
            try:
                # Use Tamil voice (PallaviNeural)
                communicate = Communicate(text=text, voice="ta-IN-PallaviNeural")
                await communicate.save(tts_file)
                
                if os.path.exists(tts_file):
                    audio_segment = AudioSegment.from_mp3(tts_file)
                    return audio_segment, tts_file
                else:
                    raise FileNotFoundError("TTS file not generated")
                    
            except Exception as e:
                logger.warning(f"TTS failed for segment {idx}: {str(e)}")
                return None, tts_file
        
        tasks = [_generate(text, i) for i, text in enumerate(texts) if text.strip()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        segments = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"TTS task failed: {result}")
                continue
                
            seg, file = result
            if seg:
                segments.append(seg)
                if file:
                    self.cleanup_files.append(file)
            elif file and os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
        
        logger.info(f"Generated {len(segments)} TTS segments")
        return segments

    async def merge_output(self, video_path: str, tts_audio: List[AudioSegment], timestamps: List[Tuple[float, float]]) -> str:
        """Combine Tamil TTS with video"""
        try:
            if not tts_audio:
                raise ValueError("No TTS audio segments to merge")
            
            # 1. Combine TTS segments with timing
            final_audio = AudioSegment.silent(duration=0)
            
            for i, segment in enumerate(tts_audio):
                if i < len(timestamps):
                    # Add silence before segment if needed
                    if i == 0:
                        silence_duration = max(0, timestamps[i][0] * 1000)  # Convert to ms
                    else:
                        prev_end = timestamps[i-1][1] * 1000
                        current_start = timestamps[i][0] * 1000
                        silence_duration = max(self.min_silence * 1000, current_start - len(final_audio))
                    
                    if silence_duration > 0:
                        final_audio += AudioSegment.silent(duration=int(silence_duration))
                    
                    final_audio += segment
            
            # 2. Export audio
            audio_out = os.path.join(self.output_dir, f"tamil_audio_{int(time.time())}.wav")
            final_audio.export(audio_out, format="wav")
            self.cleanup_files.append(audio_out)
            
            # 3. Merge with video
            output_path = os.path.join(self.output_dir, f"tamil_dubbed_{int(time.time())}.mp4")
            
            (
                ffmpeg
                .input(video_path)
                .input(audio_out)
                .output(output_path, vcodec="copy", acodec="aac", map="0:v", map="1:a", shortest=None)
                .global_args("-loglevel", "error")
                .overwrite_output()
                .run(cmd=["ffmpeg", "-nostdin"])
            )
            
            logger.info(f"Created dubbed video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Merging failed: {str(e)}")
            raise

    # ======================
    # 4. MAIN PROCESS
    # ======================
    async def process_tamil_dubbing(self, url: str, source_lang: str = "en"):
        """End-to-end Tamil dubbing pipeline"""
        if not self._is_live_stream(url):
            raise ValueError("URL is not a live stream")
        
        self.live_active = True
        logger.info("🎬 Starting Tamil dubbing (Press Ctrl+C to stop)")
        
        segment_count = 0
        
        try:
            while self.live_active:
                start_time = time.time()
                segment_count += 1
                
                logger.info(f"📥 Processing segment #{segment_count}")
                
                try:
                    # 1. Download
                    video_path = await self.download_segment(url)
                    
                    # 2. Extract audio and transcribe
                    audio_path = await self.extract_audio(video_path)
                    transcript, timestamps, detected_lang = await self.transcribe(audio_path)
                    
                    if not transcript:
                        logger.info("⏭️ No speech detected, skipping segment")
                        continue
                    
                    # 3. Translate to Tamil
                    tamil_text = await self.translate_to_tamil(transcript, source_lang or detected_lang)
                    
                    # 4. Generate Tamil TTS
                    tts_audio = await self.generate_tamil_tts(tamil_text)
                    
                    if not tts_audio:
                        logger.warning("⚠️ No TTS audio generated, skipping segment")
                        continue
                    
                    # 5. Merge
                    output_file = await self.merge_output(video_path, tts_audio, timestamps)
                    logger.info(f"✅ Tamil dubbed segment #{segment_count}: {os.path.basename(output_file)}")
                    
                    # 6. Timing control (target: segment_duration latency)
                    elapsed = time.time() - start_time
                    if elapsed < self.segment_duration:
                        sleep_time = self.segment_duration - elapsed
                        logger.info(f"⏱️ Waiting {sleep_time:.1f}s before next segment")
                        await asyncio.sleep(sleep_time)
                
                except Exception as e:
                    logger.error(f"⚠️ Segment #{segment_count} failed: {str(e)}")
                    await asyncio.sleep(3)  # Wait before retry
        
        except asyncio.CancelledError:
            logger.info("🛑 Stopped by user")
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user (Ctrl+C)")
        finally:
            self.live_active = False
            self.cleanup()

    # ======================
    # 5. HELPER FUNCTIONS
    # ======================
    def _is_live_stream(self, url: str) -> bool:
        """Check if URL is a live stream"""
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                is_live = info.get("is_live", False)
                logger.info(f"Stream status - Live: {is_live}")
                return is_live
        except Exception as e:
            logger.error(f"Failed to check live stream status: {e}")
            return False

    def cleanup(self):
        """Remove temporary files"""
        logger.info("🧹 Cleaning up temporary files...")
        cleaned = 0
        for file in self.cleanup_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    cleaned += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")
        
        logger.info(f"🧹 Cleaned up {cleaned} temporary files")
        self.cleanup_files.clear()

# ======================
# 6. EXECUTION
# ======================
async def main():
    print("🎬 Live Tamil Dubbing Tool")
    print("=" * 50)
    
    dubber = LiveTamilDubbingTool()
    
    try:
        url = input("📺 Enter YouTube Live URL: ").strip()
        if not url:
            print("❌ No URL provided")
            return
        
        source_lang = input("🌍 Source language (e.g., 'en' for English, 'auto' for auto-detect): ").strip().lower()
        if not source_lang:
            source_lang = "auto"
        
        print(f"\n🚀 Starting dubbing process...")
        print(f"📺 URL: {url}")
        print(f"🌍 Source Language: {source_lang}")
        print(f"🎯 Target Language: Tamil (ta)")
        print("=" * 50)
        
        await dubber.process_tamil_dubbing(url, source_lang if source_lang != "auto" else None)
    
    except KeyboardInterrupt:
        logger.info("🚫 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
    finally:
        dubber.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🚫 Program interrupted")
    except Exception as e:
        print(f"❌ Program error: {e}")