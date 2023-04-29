## Description

OpenVoiceOS STT plugin for [Faster Whisper](https://github.com/guillaumekln/faster-whisper)

High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model:


## Install

`pip install ovos-stt-plugin-fasterwhisper`

## Configuration

available models are `"tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v2"`

eg, to use Large model with GPU

```json
  "stt": {
    "module": "ovos-stt-plugin-fasterwhisper",
    "ovos-stt-plugin-fasterwhisper": {
        "model": "large-v2",
        "use_cuda": true,
        "compute_type": "float16",
        "beam_size": 5
    }
  }
 
```

## Models

Models will be auto downloaded by faster whisper on plugin load

