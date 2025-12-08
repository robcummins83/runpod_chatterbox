import runpod
import requests
import torchaudio
import base64
import tempfile
import os

from chatterbox.tts import ChatterboxTTS

model = None

def get_model():
    global model
    if model is None:
        model = ChatterboxTTS.from_pretrained(device="cuda")
    return model

def download_audio_file(url: str, output_path: str) -> str:
    """Download audio from a direct URL."""
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    
    # Determine extension from content-type or URL
    content_type = response.headers.get('content-type', '')
    if 'audio/mpeg' in content_type or url.endswith('.mp3'):
        ext = '.mp3'
    elif 'audio/wav' in content_type or url.endswith('.wav'):
        ext = '.wav'
    elif 'audio/mp4' in content_type or url.endswith('.m4a'):
        ext = '.m4a'
    else:
        ext = '.wav'
    
    filepath = os.path.join(output_path, f"voice_sample{ext}")
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    return filepath

def handler(event):
    try:
        input_data = event['input']
        prompt = input_data.get('prompt', '')
        audio_url = input_data.get('audio_url')  # Direct URL to audio file
        yt_url = input_data.get('yt_url')  # YouTube URL (legacy support)
        exaggeration = input_data.get('exaggeration', 0.5)
        cfg_weight = input_data.get('cfg_weight', 0.5)
        
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Track what we used for debugging
        debug_info = {
            "audio_url_received": audio_url,
            "yt_url_received": yt_url,
            "source_used": None,
            "wav_file": None
        }
        
        # Get voice sample
        if audio_url:
            wav_file = download_audio_file(audio_url, temp_dir)
            debug_info["source_used"] = "audio_url"
            debug_info["wav_file"] = wav_file
        elif yt_url:
            # Download from YouTube (legacy)
            from yt_dlp import YoutubeDL
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': 60}],
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_url])
            wav_file = os.path.join(temp_dir, 'audio.wav')
            debug_info["source_used"] = "yt_url"
            debug_info["wav_file"] = wav_file
        else:
            return {"error": "No audio source provided (need audio_url or yt_url)", "debug": debug_info}
        
        # Generate speech
        model = get_model()
        audio_tensor = model.generate(
            prompt,
            audio_prompt_path=wav_file,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        
        # Save output
        output_file = os.path.join(temp_dir, "output.wav")
        torchaudio.save(output_file, audio_tensor, model.sr)
        
        # Encode to base64
        with open(output_file, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Cleanup
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        
        return {
            "audio_base64": audio_base64,
            "sample_rate": model.sr,
            "debug": debug_info
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
