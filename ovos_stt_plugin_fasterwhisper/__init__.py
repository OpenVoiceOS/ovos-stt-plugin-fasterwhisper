import numpy as np
from faster_whisper import WhisperModel, decode_audio, available_models
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.templates.transformers import AudioLanguageDetector
from ovos_utils.log import LOG
from speech_recognition import AudioData


class FasterWhisperLangClassifier(AudioLanguageDetector):
    def __init__(self, config=None):
        config = config or {}
        super().__init__("ovos-audio-transformer-plugin-fasterwhisper", 10, config)
        model = self.config.get("model") or "small"
        valid_model = model in FasterWhisperSTT.MODELS
        if not valid_model:
            LOG.info(f"{model} is not default model_id ({FasterWhisperSTT.MODELS}), "
                     f"assuming huggingface repo_id or path to local model")

        self.compute_type = self.config.get("compute_type", "int8")
        self.use_cuda = self.config.get("use_cuda", False)
        self.beam_size = self.config.get("beam_size", 5)
        self.cpu_threads = self.config.get("cpu_threads", 4)

        if self.use_cuda:
            device = "cuda"
        else:
            device = "cpu"
        self.engine = WhisperModel(model, device=device, compute_type=self.compute_type)

    @staticmethod
    def audiochunk2array(audio_data: bytes):
        # Convert buffer to float32 using NumPy
        audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2 ** 15
        data = audio_as_np_float32 / max_int16
        return data

    def detect(self, audio_data: bytes, valid_langs=None):
        if isinstance(audio_data, AudioData):
            audio_data = audio_data.get_wav_data()
        valid_langs = [l.lower().split("-")[0] for l in valid_langs or self.valid_langs]
        audio = self.audiochunk2array(audio_data)
        if not self.engine.model.is_multilingual:
            language = "en"
            language_probability = 1
        else:
            sampling_rate = self.engine.feature_extractor.sampling_rate

            if not isinstance(audio, np.ndarray):
                audio = decode_audio(audio, sampling_rate=sampling_rate)

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
        model = self.config.get("model") or "whisper-large-v3-turbo"
        if model == "whisper-large-v3-turbo":
            model = "deepdml/faster-whisper-large-v3-turbo-ct2"
        else:
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

    @staticmethod
    def audiodata2array(audio_data):
        assert isinstance(audio_data, AudioData)
        return FasterWhisperLangClassifier.audiochunk2array(audio_data.get_wav_data())

    def execute(self, audio, language=None):
        lang = language or self.lang
        segments, _ = self.engine.transcribe(
            self.audiodata2array(audio),
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            language=lang.split("-")[0].lower(),
            vad_filter = self.config.get("vad_filter", False)
        )
        # segments is an iterator, transcription only happens here
        transcription = "".join(segment.text for segment in segments).strip()
        return transcription

    @property
    def available_languages(self) -> set:
        return set(FasterWhisperSTT.LANGUAGES.keys())


FasterWhisperSTTConfig = {
    lang: [
        {
            "model": "tiny",
            "lang": lang,
            "meta": {
                "priority": 50,
                "display_name": "FasterWhisper (Tiny)",
                "offline": True,
            },
        },
        {
            "model": "base",
            "lang": lang,
            "meta": {
                "priority": 55,
                "display_name": f"FasterWhisper (Base)",
                "offline": True,
            },
        },
        {
            "model": "small",
            "lang": lang,
            "meta": {
                "priority": 60,
                "display_name": f"FasterWhisper (Small)",
                "offline": True,
            },
        },
    ]
    for lang, lang_name in FasterWhisperSTT.LANGUAGES.items()
}

if __name__ == "__main__":
    b = FasterWhisperSTT(config={"model": "projecte-aina/faster-whisper-large-v3-ca-3catparla"})

    from speech_recognition import Recognizer, AudioFile

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-vosk/example.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    a = b.execute(audio, language="ca")
    print(a)
    # And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.

    l = FasterWhisperLangClassifier()
    lang, prob = l.detect(audio.get_wav_data(),
                          valid_langs=["pt", "es", "ca", "gl"])
    print(lang, prob)
    # es 0.7143379217828251
