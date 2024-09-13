#!/bin/bash

# Exit on any error
set -e

# Function to display help
show_help() {
    echo "Usage: bash scripts/run.sh [options] input_audio_file"
    echo ""
    echo "Transcribe an audio file using OpenAI's Whisper model."
    echo ""
    echo "Positional arguments:"
    echo "  input_audio_file          Path to the input audio file to be transcribed."
    echo ""
    echo "Optional arguments:"
    echo "  --output_audio OUTPUT_AUDIO    Path to save the preprocessed audio file."
    echo "  --output_text OUTPUT_TEXT      Path to save the transcription text file."
    echo "  -h, --help                     Show this help message and exit."
    echo ""
    echo "Example:"
    echo "  bash scripts/run.sh examples/sample_audio.m4a --output_text results/transcription.txt"
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" || "$#" -eq 0 ]]; then
    show_help
    exit 0
fi

# Activate the Python environment
source env/bin/activate

# Run the transcription script with any arguments passed to this script
python src/transcribe.py "$@"

# Deactivate the environment when done
deactivate
