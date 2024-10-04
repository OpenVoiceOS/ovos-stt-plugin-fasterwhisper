## Description

OpenVoiceOS STT plugin for [Faster Whisper](https://github.com/guillaumekln/faster-whisper)

High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model:


## Install

`pip install ovos-stt-plugin-fasterwhisper`

## Models

available models are `'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'distil-large-v2', 'distil-medium.en', 'distil-small.en', 'distil-large-v3'`

you can also pass a full path to a local model or a huggingface repo_id, eg. `"projecte-aina/faster-whisper-large-v3-ca-3catparla"`

You can [convert](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#model-conversion) any whisper model, or use any [compatible model from huggingface](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=modified&search=faster-whisper)

## Configuration

to use Large model with GPU

```json
  "stt": {
    "module": "ovos-stt-plugin-fasterwhisper",
    "ovos-stt-plugin-fasterwhisper": {
        "model": "large-v3",
        "use_cuda": true,
        "compute_type": "float16",
        "beam_size": 5,
        "cpu_threads": 4
    }
  }
```

To use Whisper for lang detection (ovos-dinkum-listener only)


```json
  "listener": {
    "audio_transformers": {
        "ovos-audio-transformer-plugin-fasterwhisper": {
            "model": "small"
        }
    }
  }
```

