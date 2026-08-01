from typing import Tuple, List, Optional

from faster_whisper import WhisperModel, available_models
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.templates.transformers import AudioLanguageDetector
from ovos_plugin_manager.utils.audio import AudioData, AudioFile
from ovos_utils import classproperty
from ovos_utils.log import LOG


class FasterWhisperLangClassifier(AudioLanguageDetector):
    def __init__(self, config=None):
        config = config or {"model": "small"}
        super().__init__("ovos-audio-transformer-plugin-fasterwhisper", 10, config)
        self.engine = FasterWhisperSTT(config=config)

    def detect(self, audio_data: bytes, valid_langs=None, sample_rate=16000, sample_width=2) -> Tuple[str, float]:
        if not isinstance(audio_data, AudioData):
            audio_data = AudioData(audio_data,
                                   sample_rate=sample_rate,
                                   sample_width=sample_width)
        return self.engine.detect_language(audio_data, valid_langs)


class FasterWhisperSTT(STT):
    MODELS = available_models()
    LANGUAGES = {
        "en": "english",
        "zh": "chinese",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ko": "korean",
        "fr": "french",
        "ja": "japanese",
        "pt": "portuguese",
        "tr": "turkish",
        "pl": "polish",
        "ca": "catalan",
        "nl": "dutch",
        "ar": "arabic",
        "sv": "swedish",
        "it": "italian",
        "id": "indonesian",
        "hi": "hindi",
        "fi": "finnish",
        "vi": "vietnamese",
        "iw": "hebrew",
        "uk": "ukrainian",
        "el": "greek",
        "ms": "malay",
        "cs": "czech",
        "ro": "romanian",
        "da": "danish",
        "hu": "hungarian",
        "ta": "tamil",
        "no": "norwegian",
        "th": "thai",
        "ur": "urdu",
        "hr": "croatian",
        "bg": "bulgarian",
        "lt": "lithuanian",
        "la": "latin",
        "mi": "maori",
        "ml": "malayalam",
        "cy": "welsh",
        "sk": "slovak",
        "te": "telugu",
        "fa": "persian",
        "lv": "latvian",
        "bn": "bengali",
        "sr": "serbian",
        "az": "azerbaijani",
        "sl": "slovenian",
        "kn": "kannada",
        "et": "estonian",
        "mk": "macedonian",
        "br": "breton",
        "eu": "basque",
        "is": "icelandic",
        "hy": "armenian",
        "ne": "nepali",
        "mn": "mongolian",
        "bs": "bosnian",
        "kk": "kazakh",
        "sq": "albanian",
        "sw": "swahili",
        "gl": "galician",
        "mr": "marathi",
        "pa": "punjabi",
        "si": "sinhala",
        "km": "khmer",
        "sn": "shona",
        "yo": "yoruba",
        "so": "somali",
        "af": "afrikaans",
        "oc": "occitan",
        "ka": "georgian",
        "be": "belarusian",
        "tg": "tajik",
        "sd": "sindhi",
        "gu": "gujarati",
        "am": "amharic",
        "yi": "yiddish",
        "lo": "lao",
        "uz": "uzbek",
        "fo": "faroese",
        "ht": "haitian creole",
        "ps": "pashto",
        "tk": "turkmen",
        "nn": "nynorsk",
        "mt": "maltese",
        "sa": "sanskrit",
        "lb": "luxembourgish",
        "my": "myanmar",
        "bo": "tibetan",
        "tl": "tagalog",
        "mg": "malagasy",
        "as": "assamese",
        "tt": "tatar",
        "haw": "hawaiian",
        "ln": "lingala",
        "ha": "hausa",
        "ba": "bashkir",
        "jw": "javanese",
        "su": "sundanese",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = self.config.get("model") or "large-v3-turbo"
        valid_model = model in FasterWhisperSTT.MODELS
        if not valid_model:
            LOG.info(f"{model} is not default model_id ({FasterWhisperSTT.MODELS}), "
                     f"assuming huggingface repo_id or path to local model")

        self.beam_size = self.config.get("beam_size", 5)
        self.compute_type = self.config.get("compute_type", "int8")
        self.use_cuda = self.config.get("use_cuda", False)
        self.cpu_threads = self.config.get("cpu_threads", 4)

        if self.use_cuda:
            device = "cuda"
        else:
            device = "cpu"
        self.engine = WhisperModel(
            model,
            device=device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )


    def detect_language(self, audio: AudioData, valid_langs: Optional[List[str]] = None) -> Tuple[str, float]:
        valid_langs = [l.lower().split("-")[0]
                       for l in valid_langs or self.available_languages]
        audio = audio.get_np_float32(self.engine.feature_extractor.sampling_rate)
        if not self.engine.model.is_multilingual:
            language = "en"
            language_probability = 1
        else:
            features = self.engine.feature_extractor(audio)
            segment = features[:, : self.engine.feature_extractor.nb_max_frames]
            encoder_output = self.engine.encode(segment)
            results = self.engine.model.detect_language(encoder_output)[0]
            results = [(l[2:-2], p) for l, p in results if l[2:-2] in valid_langs]
            total = sum(l[1] for l in results) or 1
            results = sorted(
                [(l, p / total) for l, p in results], key=lambda k: k[1], reverse=True
            )
            language, language_probability = results[0]
        return language, language_probability

    def execute(self, audio: AudioData, language=None):
        lang = language or self.lang
        if lang == "auto":
            lang, _ = self.detect_language(audio)
        segments, _ = self.engine.transcribe(
            audio.get_np_float32(self.engine.feature_extractor.sampling_rate),
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            language=lang.split("-")[0].lower(),
            vad_filter=self.config.get("vad_filter", False)
        )
        # segments is an iterator, transcription only happens here
        transcription = "".join(segment.text for segment in segments).strip()
        return transcription

    @classproperty
    def available_languages(cls) -> set:
        return set(cls.LANGUAGES.keys())


FasterWhisperSTTConfig = {
    lang: [{
        "model": model,
        "lang": lang,
        "meta": {
            "priority": 50,
            "display_name": f"FasterWhisper ({model})",
            "offline": True,
        },
    } for model in FasterWhisperSTT.MODELS]
    for lang in FasterWhisperSTT.available_languages
}

if __name__ == "__main__":
    print(FasterWhisperSTT.MODELS)

    b = FasterWhisperSTT(config={
        "model": "large-v3-turbo",
        "use_cuda": True,
        "compute_type": "float16",
        "beam_size": 5,
        "cpu_threads": 8
    })

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = source.read()

    a = b.execute(audio, language="en")
    print(a)
    # And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
    print(b.detect_language(audio))

    l = FasterWhisperLangClassifier()
    lang, prob = l.detect(audio.get_raw_data(),
                          valid_langs=["en", "pt", "es", "ca", "gl"])
    print(lang, prob)
