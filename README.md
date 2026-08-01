## Description

This plugin adds [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) speech recognition to OpenVoiceOS. Faster Whisper is a fast inference engine for [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition models.

The plugin also provides a language detection transformer for `ovos-dinkum-listener`.

## Install

```bash
pip install ovos-stt-plugin-fasterwhisper
```

## Models

The plugin supports these Whisper model names: `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large`, `distil-large-v2`, `distil-medium.en`, `distil-small.en`, `distil-large-v3`.

You can also pass a full path to a local model, or a Hugging Face repo ID, for example `projecte-aina/faster-whisper-large-v3-ca-3catparla`.

To use a model that is not already in Faster Whisper format, [convert it](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#model-conversion), or pick a [compatible model on Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=modified&search=faster-whisper).

## Configuration

This example configures the `large-v3` model to run on GPU:

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

This example uses Faster Whisper for language detection. It works only with `ovos-dinkum-listener`.

```json
  "listener": {
    "audio_transformers": {
        "ovos-audio-transformer-plugin-fasterwhisper": {
            "model": "small"
        }
    }
  }
```

## Related projects

- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) — the plugin manager that loads STT and audio transformer plugins.
- [OpenVoiceOS/ovos-dinkum-listener](https://github.com/OpenVoiceOS/ovos-dinkum-listener) — the listener service that runs audio transformer plugins such as the language detector in this repository.

## License

Apache-2.0
