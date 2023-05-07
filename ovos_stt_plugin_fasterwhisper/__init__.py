# this is needed to read the WAV file properly
import numpy as np
from faster_whisper import WhisperModel, decode_audio
from ovos_bus_client.session import SessionManager
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
            model = "small"

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
        audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2 ** 15
        data = audio_as_np_float32 / max_int16
        return data

    def detect(self, audio, valid_langs=None):
        if not valid_langs:
            s = SessionManager.get()
            valid_langs = s.valid_languages
        valid_langs = [l.lower().split("-")[0] for l in valid_langs]

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
            results = sorted([(l, p / total) for l, p in results], key=lambda k: k[1], reverse=True)

            language, language_probability = results[0]
        return language, language_probability

    # plugin api
    def transform(self, audio_data):
        lang, prob = self.detect(self.audiochunk2array(audio_data))
        LOG.info(f"Detected speech language '{lang}' with probability {prob}")
        return audio_data, {"stt_lang": lang, "lang_probability": prob}


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
            model = "small"
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
        lang = language or self.lang
        segments, _ = self.engine.transcribe(self.audiodata2array(audio), beam_size=self.beam_size,
                                             condition_on_previous_text=False, language=lang.split("-")[0].lower())
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
