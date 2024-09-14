## Description

OpenVoiceOS STT plugin for [Faster Whisper](https://github.com/guillaumekln/faster-whisper)

High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model:


## Install

`pip install ovos-stt-plugin-fasterwhisper`

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

## Models

available models are `'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'distil-large-v2', 'distil-medium.en', 'distil-small.en', 'distil-large-v3'`

you can also pass a full path to a local model or a huggingface repo_id,
eg. `"projecte-aina/faster-whisper-large-v3-ca-3catparla"`

You can [convert](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#model-conversion) any whisper model, or use any [compatible model from huggingface](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=modified&search=faster-whisper)

### Portuguese

self reported WER score from model pages

| Model                             | CV13   | Fleurs |
|-----------------------------------|--------|--------|
| `my-north-ai/whisper-large-v3-pt` | ?      | 4.86   |
| `my-north-ai/whisper-medium-pt`   | ?      | 6.97   |
| `my-north-ai/whisper-small-pt`    | ?      | 10.9   |
| `zuazo/whisper-large-v3-pt`       | 4.6003 | ?      |
| `zuazo/whisper-large-v2-pt`       | 5.875  | ?      |
| `zuazo/whisper-large-pt`          | 6.399  | ?      |
| `zuazo/whisper-medium-pt`         | 6.332  | ?      |
| `zuazo/whisper-small-pt`          | 10.252 | ?      |
| `zuazo/whisper-base-pt`           | 19.290 | ?      |
| `zuazo/whisper-tiny-pt`           | 28.965 | ?      |

### Galician

self reported WER score from model pages

| Model                       | CV13    |
|-----------------------------|---------|
| `zuazo/whisper-large-v2-gl` | 5.9879  |
| `zuazo/whisper-large-gl`    | 6.9398  |
| `zuazo/whisper-medium-gl`   | 7.1227  |
| `zuazo/whisper-small-gl`    | 10.9875 |
| `zuazo/whisper-base-gl`     | 18.6879 |
| `zuazo/whisper-tiny-gl`     | 26.3504 |

### Catalan

self reported WER score from model pages

| Model                                                | CV13    | CV17 (test) | 3CatParla (test) |
|------------------------------------------------------|---------|-------------|------------------|
| `projecte-aina/faster-whisper-large-v3-ca-3catparla` | ?       | 9.260       | 0.960            |
| `zuazo/whisper-large-v2-ca`                          | 4.6716  | ?           |                  |
| `zuazo/whisper-large-ca`                             | 5.0700  | ?           |                  |
| `zuazo/whisper-large-v3-ca`                          | 5.9714  | ?           |                  |
| `zuazo/whisper-medium-ca`                            | 5.9954  | ?           |                  |
| `zuazo/whisper-small-ca`                             | 10.0252 | ?           |                  |
| `zuazo/whisper-base-ca`                              | 13.7897 | ?           |                  |
| `zuazo/whisper-tiny-ca`                              | 16.9043 | ?           |                  |

### Spanish

self reported WER score from model pages

| Model                       | CV13    |
|-----------------------------|---------|
| `zuazo/whisper-large-v2-es` | 4.8949  |
| `zuazo/whisper-large-es`    | 5.1265  |
| `zuazo/whisper-medium-es`   | 5.4088  |
| `zuazo/whisper-small-es`    | 8.2668  |
| `zuazo/whisper-base-es`     | 13.5312 |
| `zuazo/whisper-tiny-es`     | 19.5904 |

### Basque

self reported WER score from model pages

| Model                              | CV13    |
|------------------------------------|---------|
| `zuazo/whisper-large-v3-eu-cv16_1` | 6.8880  |
| `zuazo/whisper-large-v2-eu-cv16_1` | 7.7204  |
| `zuazo/whisper-large-eu-cv16_1`    | 8.1444  |
| `zuazo/whisper-medium-eu-cv16_1`   | 9.2006  |
| `zuazo/whisper-small-eu-cv16_1`    | 12.7374 |
| `zuazo/whisper-base-eu-cv16_1`     | 16.1765 |
| `zuazo/whisper-tiny-eu-cv16_1`     | 19.0949 |
