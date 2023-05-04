# this is needed to read the WAV file properly
import numpy
from faster_whisper import WhisperModel
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovos_utils.log import LOG
from speech_recognition import AudioData


class FasterWhisperLangClassifier(AudioTransformer):
    def __init__(self, config=None):
        config = config or {}
        super().__init__("ovos-audio-transformer-plugin-fasterwhisper", 10, config)
        model = self.config.get("model")
        if not model:
            model = "small.en"
        assert model in FasterWhisperSTT.MODELS  # TODO - better error handling

        self.compute_type = self.config.get("compute_type", "int8")
        self.use_cuda = self.config.get("use_cuda", False)
        self.beam_size = self.config.get("beam_size", 5)

        if self.use_cuda:
            device = "cuda"
        else:
            device = "cpu"
        self.engine = WhisperModel(model, device=device, compute_type=self.compute_type)

    @staticmethod
    def audiochunk2array(audio_data):
        # Convert buffer to float32 using NumPy
        audio_as_np_int16 = numpy.frombuffer(audio_data, dtype=numpy.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(numpy.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2 ** 15
        data = audio_as_np_float32 / max_int16
        return data

    # plugin api
    def transform(self, audio_data):
        # segments is an iterator, transcription is not done here
        _, info = self.engine.transcribe(self.audiochunk2array(audio_data), beam_size=self.beam_size)
        LOG.info(f"Detected speech language '{info.language}' with probability {info.language_probability}")
        return audio_data, {"stt_lang": info.language, "lang_probability": info.language_probability}


class FasterWhisperSTT(STT):
    MODELS = ("tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large", "large-v2")
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
        model = self.config.get("model")
        if not model:
            model = "small.en"
        assert model in self.MODELS  # TODO - better error handling

        self.beam_size = self.config.get("beam_size", 5)
        self.compute_type = self.config.get("compute_type", "int8")
        self.use_cuda = self.config.get("use_cuda", False)

        if self.use_cuda:
            device = "cuda"
        else:
            device = "cpu"
        self.engine = WhisperModel(model, device=device, compute_type=self.compute_type)

    @staticmethod
    def audiodata2array(audio_data):
        assert isinstance(audio_data, AudioData)
        return FasterWhisperLangClassifier.audiochunk2array(audio_data.get_wav_data())

    def execute(self, audio, language=None):
        segments, _ = self.engine.transcribe(self.audiodata2array(audio), beam_size=self.beam_size)
        # segments is an iterator, transcription only happens here
        transcription = "".join(segment.text for segment in segments).strip()
        return transcription

    @property
    def available_languages(self) -> set:
        return set(FasterWhisperSTT.LANGUAGES.keys())


FasterWhisperSTTConfig = {
    lang: [{"model": "tiny",
            "lang": lang,
            "meta": {
                "priority": 50,
                "display_name": f"FasterWhisper (Tiny)",
                "offline": True}
            },
           {"model": "base",
            "lang": lang,
            "meta": {
                "priority": 55,
                "display_name": f"FasterWhisper (Base)",
                "offline": True}
            },
           {"model": "small",
            "lang": lang,
            "meta": {
                "priority": 60,
                "display_name": f"FasterWhisper (Small)",
                "offline": True}
            }
           ]
    for lang, lang_name in FasterWhisperSTT.LANGUAGES.items()
}

if __name__ == "__main__":
    b = FasterWhisperSTT()
    from speech_recognition import Recognizer, AudioFile

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    a = b.execute(audio, language="en")
    # 2023-04-29 17:42:30.769 - OVOS - __main__:execute:145 - INFO - Detected speech language 'en' with probability 1
    print(a)
    # And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
