# tk-llm

A Python GUI launcher for local LLM models using llama.cpp. Features a user-friendly tkinter interface for running GGUF models like GPT-OSS 20B and DeepSeek Coder with customizable parameters.

## Features
- Multi-model support with automatic format detection
- Real-time output streaming in GUI or terminal mode
- Adjustable generation parameters (temperature, top-k, top-p)
- System/user prompt management
- Configuration save/load functionality
- Support for Harmony Response Format and simple prompting

## Requirements
- Python 3.x with tkinter
- llama-cli executable from llama.cpp
- Compatible GGUF model files

## Usage
```bash
python prompt.py
```

Place model files in the same directory as the script.
