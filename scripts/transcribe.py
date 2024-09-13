#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from tqdm import tqdm
import librosa
import ffmpeg

def check_and_install_dependencies():
    """
    Checks for missing dependencies and installs them.
    """
    required_packages = [
        "transformers", "datasets[audio]", "accelerate", "torch", "ffmpeg-python", "tqdm", "librosa"
    ]

    for package in required_packages:
        try:
            __import__(package.split("[")[0])  # Handle cases like 'datasets[audio]'
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Install flash-attn if CUDA is available
    if torch.cuda.is_available():
        cuda_home = os.environ.get('CUDA_HOME') or find_cuda_home()
        if cuda_home:
            os.environ['CUDA_HOME'] = cuda_home
            os.environ['PATH'] = f"{cuda_home}/bin:" + os.environ['PATH']
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_home}/lib64:" + os.environ.get('LD_LIBRARY_PATH', '')
            print(f"CUDA_HOME set to {cuda_home}")
        else:
            print("CUDA installation not found.")
            try:
                # Automatically check and install CUDA using install_cuda_nvcc.py
                subprocess.check_call([sys.executable, "src/install_cuda_nvcc.py"])
            except subprocess.CalledProcessError as e:
                print("CUDA installation failed.")
                sys.exit(1)

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"])
        except subprocess.CalledProcessError:
            print("Failed to install flash-attn. Skipping it.")

def find_cuda_home():
    """
    Finds the CUDA installation path.
    """
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version'], universal_newlines=True)
        for line in nvcc_output.split('\n'):
            if 'release' in line:
                version = line.split('release ')[1].split(',')[0]
                cuda_path = f"/usr/local/cuda-{version}"
                if os.path.exists(cuda_path):
                    return cuda_path
        # If the above fails, try to find any cuda directory
        cuda_dirs = [d for d in os.listdir('/usr/local') if d.startswith('cuda-')]
        if cuda_dirs:
            return os.path.join('/usr/local', max(cuda_dirs))
    except FileNotFoundError:
        print("nvcc not found. CUDA might not be installed correctly.")
    return None

def preprocess_audio(input_file, output_file):
    """
    Preprocess the audio file by removing silence and converting it to WAV format.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the preprocessed audio file.

    Returns:
        str: Path to the preprocessed audio file, or None if preprocessing failed.
    """
    print(f"Preprocessing audio: {input_file}")
    try:
        output_file = os.path.splitext(output_file)[0] + '.wav'
        (
            ffmpeg
            .input(input_file)
            .output(output_file, af="silenceremove=1:0:-50dB", format='wav')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Preprocessed audio saved as: {output_file}")
        return output_file
    except ffmpeg.Error as e:
        print(f"Failed to process audio: {e}")
        return None

def run_transcription(input_audio, output_text, device):
    """
    Transcribe the preprocessed audio file using the Whisper model.

    Args:
        input_audio (str): Path to the preprocessed audio file.
        output_text (str): Path to save the transcription text.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        bool: True if transcription succeeded, False otherwise.
    """
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "openai/whisper-large-v3"
    print("Loading model...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    processor = WhisperProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=device,
    )

    generate_kwargs = {
        "max_new_tokens": 445,
        "num_beams": 1,
        "temperature": 0.0,
    }

    audio_duration = librosa.get_duration(path=input_audio)

    print(f"Transcribing: {input_audio}")
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            with tqdm(total=int(audio_duration), desc="Transcription Progress", unit="sec") as pbar:
                result = pipe(
                    input_audio,
                    generate_kwargs=generate_kwargs,
                    return_timestamps=True,
                    chunk_length_s=30,
                    stride_length_s=[5, 5],
                )
                pbar.update(int(audio_duration))

            transcription_text = result["text"]
            print(f"Transcription:\n{transcription_text}")

            with open(output_text, "w") as f:
                f.write(transcription_text)
            print(f"Transcription saved to: {output_text}")

            return True
        except Exception as e:
            print(f"Error during transcription: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying transcription ({retry_count}/{max_retries})...")
            else:
                print("Max retries reached. Transcription failed.")
                return False

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI's Whisper model.")
    parser.add_argument('input_audio', type=str, help='Path to the input audio file.')
    parser.add_argument('--output_audio', type=str, help='Path to save the preprocessed audio file.')
    parser.add_argument('--output_text', type=str, help='Path to save the transcription text file.')
    args = parser.parse_args()

    input_audio = args.input_audio
    output_audio = args.output_audio or os.path.splitext(input_audio)[0] + '_nosilence.wav'
    output_text = args.output_text or os.path.splitext(input_audio)[0] + '_transcription.txt'

    if not os.path.exists(input_audio):
        print(f"Input audio file not found: {input_audio}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check and install necessary dependencies
    check_and_install_dependencies()

    # Step 1: Preprocess audio
    output_audio = preprocess_audio(input_audio, output_audio)
    if not output_audio:
        print("Audio preprocessing failed. Exiting.")
        sys.exit(1)

    # Step 2: Transcribe
    success = run_transcription(output_audio, output_text, device)

    if success:
        print("Transcription completed successfully.")
    else:
        print("Transcription failed.")

if __name__ == "__main__":
    main()
