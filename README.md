# 🎬 Live Tamil Dubbing Tool

A real-time YouTube live stream dubbing tool that automatically translates and dubs live content into Tamil using AI-powered speech recognition, translation, and text-to-speech synthesis.

## ✨ Features

- **Real-time Processing**: Processes live YouTube streams with ~15-second latency
- **Multi-language Support**: Auto-detects source language and translates to Tamil
- **High-quality TTS**: Uses Microsoft Edge Tamil neural voices
- **GPU Acceleration**: Supports CUDA for faster Whisper transcription
- **Automatic Cleanup**: Manages temporary files automatically
- **Error Recovery**: Robust error handling with automatic retries

## 🛠️ Prerequisites

### System Requirements
- Python 3.8 or higher
- FFmpeg installed on your system
- Internet connection for translation and TTS services
- Optional: NVIDIA GPU with CUDA for faster processing

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

## 📦 Installation

1. **Clone or download the script:**
```bash
# Save the live_tamil_dubbing.py file to your desired directory
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv tamil_dubbing_env
source tamil_dubbing_env/bin/activate  # Linux/macOS
# OR
tamil_dubbing_env\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **For GPU support (optional but recommended):**
```bash
# Install CUDA-enabled PyTorch (check https://pytorch.org for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Usage

1. **Run the tool:**
```bash
python live_tamil_dubbing.py
```

2. **Provide inputs when prompted:**
   - **YouTube Live URL**: Enter the live stream URL (e.g., `https://www.youtube.com/watch?v=LIVE_VIDEO_ID`)
   - **Source Language**: Enter the source language code (e.g., `en` for English, or `auto` for auto-detection)

3. **Monitor the process:**
   - The tool will start processing segments every 15 seconds
   - Watch the console for progress updates
   - Dubbed video segments will be saved in the `tamil_dubbed_output/` directory

4. **Stop the process:**
   - Press `Ctrl+C` to stop the dubbing process
   - The tool will clean up temporary files automatically

## 📁 Output Structure

```
tamil_dubbed_output/
├── tamil_dubbed_1703123456.mp4    # Final dubbed video segments
├── tamil_dubbed_1703123471.mp4
└── ...
```

## ⚙️ Configuration

You can modify these settings in the `LiveTamilDubbingTool` class:

- `segment_duration`: Processing interval in seconds (default: 15)
- `min_silence`: Minimum silence gap between audio segments (default: 0.1s)
- Whisper model size: Change `"small"` to `"tiny"`, `"base"`, `"medium"`, or `"large"`

## 🌍 Supported Languages

### Source Languages (Auto-detected)
- English (en)
- Hindi (hi)
- Spanish (es)
- French (fr)
- German (de)
- Japanese (ja)
- And many more supported by Whisper

### Target Language
- Tamil (ta) - Uses Microsoft Edge Tamil neural voices

## 🔧 Troubleshooting

### Common Issues

**1. "FFmpeg not found" error:**
```bash
# Make sure FFmpeg is installed and in your PATH
ffmpeg -version
```

**2. "No module named 'torch'" error:**
```bash
# Install PyTorch
pip install torch torchaudio
```

**3. "Download failed" error:**
- Check if the YouTube URL is a live stream
- Verify your internet connection
- Some streams may have restrictions

**4. "CUDA out of memory" error:**
```python
# In the code, change the Whisper model to a smaller one:
self.model = whisper.load_model("tiny", device=self.device)  # Instead of "small"
```

**5. Translation API limits:**
- The tool uses Google Translate which may have rate limits
- Consider adding delays between requests for heavy usage

### Performance Tips

1. **Use GPU acceleration:**
   - Install CUDA-enabled PyTorch for 3-5x faster transcription

2. **Adjust segment duration:**
   - Increase `segment_duration` for less frequent processing
   - Decrease for lower latency (but higher resource usage)

3. **Use smaller Whisper models:**
   - `tiny`: Fastest, lower accuracy
   - `small`: Good balance (default)
   - `base/medium/large`: Higher accuracy, slower

## 📝 Example Usage

```bash
$ python live_tamil_dubbing.py
🎬 Live Tamil Dubbing Tool
==================================================
📺 Enter YouTube Live URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
🌍 Source language (e.g., 'en' for English, 'auto' for auto-detect): en

🚀 Starting dubbing process...
📺 URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
🌍 Source Language: en
🎯 Target Language: Tamil (ta)
==================================================
2024-01-01 12:00:00 - INFO - Using device: cuda
2024-01-01 12:00:01 - INFO - Whisper model loaded successfully
2024-01-01 12:00:02 - INFO - 🎬 Starting Tamil dubbing (Press Ctrl+C to stop)
...
```

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.

## ⚠️ Disclaimer

- This tool is for educational and personal use only
- Respect YouTube's Terms of Service and copyright laws
- Some live streams may have access restrictions
- Translation and TTS quality may vary depending on content complexity

## 📄 License

This project is open source. Please ensure you comply with the licenses of all dependencies used.