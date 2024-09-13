Whisper Large V3 Transcription
Description
This project utilizes OpenAI's Whisper Large V3 model to transcribe audio files efficiently, even on laptops with GPUs like the NVIDIA RTX 3070 Ti with 8GB VRAM. The project includes scripts for setting up the necessary environment, installing dependencies, and processing audio files for transcription.

Features
CUDA Support: Optimized for NVIDIA GPUs with CUDA capabilities.
Audio Preprocessing: Removes silence from audio files to improve transcription accuracy.
Efficient Transcription: Utilizes the Whisper Large V3 model with support for low-memory GPU setups.
User-Friendly Scripts: Easy setup and execution with provided scripts.
Automated Dependency Checks: Ensures all necessary packages and system requirements are met.
Table of Contents
Whisper Large V3 Transcription
Description
Features
Table of Contents
Dependencies
Project Structure
Setup Instructions
1. Clone the Repository
2. Install CUDA and NVIDIA Drivers
3. Run the Setup Script
4. Activate the Virtual Environment
Usage
Transcribe an Audio File
Example
Display Help Information
Contributing
License
Acknowledgements
Dependencies
Python 3.8+

CUDA Toolkit and NVIDIA Drivers (for GPU acceleration)

Python Packages:

Refer to requirements.txt for specific versions.

torch
transformers
datasets[audio]
librosa
ffmpeg-python
tqdm
accelerate
ffmpeg (system package)
Project Structure
bash
Copy code
/project-root
│
├── env/                     # Virtual environment directory (not included in Git)
│
├── src/                     # Source files
│   ├── transcribe.py                 # Main Python script for transcription
│   └── install_cuda_nvcc.py          # Script to install CUDA and NVCC
│
├── scripts/                 # Shell scripts
│   ├── setup.sh                     # Setup script for installing dependencies
│   └── run.sh                       # Script to run the transcription
│
├── examples/                # Example audio files
│   └── sample_audio.m4a
│
├── results/                 # Directory for output results
│   └── (transcription outputs)
│
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
Setup Instructions
1. Clone the Repository
bash
Copy code
git clone [repository-url]
cd project-root
2. Install CUDA and NVIDIA Drivers
Ensure you have a CUDA-capable NVIDIA GPU installed.

Verify CUDA-Capable GPU
bash
Copy code
lspci | grep -i nvidia
If no output is returned, update the PCI hardware database:

bash
Copy code
sudo update-pciids
lspci | grep -i nvidia
Refer to CUDA GPUs to confirm your GPU is supported.

Install CUDA Toolkit and Drivers
Follow the instructions specific to your Linux distribution:

Ubuntu 24.04

The install_cuda_nvcc.py script can automatically install CUDA for Ubuntu 24.04.

Other Distributions

Refer to the NVIDIA CUDA Downloads page for instructions specific to your distribution.

3. Run the Setup Script
This script creates a Python virtual environment and installs all necessary Python dependencies.

bash
Copy code
bash scripts/setup.sh
4. Activate the Virtual Environment
bash
Copy code
source env/bin/activate
Usage
Transcribe an Audio File
Use the run.sh script to transcribe an audio file.

bash
Copy code
bash scripts/run.sh [options] input_audio_file
Example
bash
Copy code
bash scripts/run.sh examples/sample_audio.m4a --output_text results/transcription.txt
Display Help Information
bash
Copy code
bash scripts/run.sh --help
Output:

vbnet
Copy code
Usage: bash scripts/run.sh [options] input_audio_file

Transcribe an audio file using OpenAI's Whisper model.

Positional arguments:
  input_audio_file              Path to the input audio file to be transcribed.

Optional arguments:
  --output_audio OUTPUT_AUDIO   Path to save the preprocessed audio file.
  --output_text OUTPUT_TEXT     Path to save the transcription text file.
  -h, --help                    Show this help message and exit.

Example:
  bash scripts/run.sh examples/sample_audio.m4a --output_text results/transcription.txt
Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

Steps to Contribute:

Fork the project repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them with clear messages.
Push your changes to your forked repository.
Submit a pull request to the main repository.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
OpenAI for the Whisper model.
Hugging Face Transformers for the Transformers library.
PyTorch for the deep learning framework.
FFmpeg for audio processing capabilities.
Note: Ensure that all system dependencies like CUDA, NVIDIA drivers, and FFmpeg are properly installed and configured on your system before running the transcription script.

For any issues or questions, please open an issue on the GitHub repository or contact the maintainer.

