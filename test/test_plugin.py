import numpy as np
import pytest
from speech_recognition import AudioFile, Recognizer

from ovos_stt_plugin_fasterwhisper import FasterWhisperLangClassifier, FasterWhisperSTT


@pytest.fixture
def audio_data():
    recognizer = Recognizer()
    with AudioFile("jfk.wav") as source:
        return recognizer.record(source)


def test_faster_whisper_stt_execute(audio_data):
    stt = FasterWhisperSTT()
    transcription = stt.execute(audio_data, language="en")
    assert isinstance(transcription, str)
    assert len(transcription) > 0


def test_faster_whisper_stt_available_languages():
    stt = FasterWhisperSTT()
    available_languages = stt.available_languages
    assert isinstance(available_languages, set)
    assert "en" in available_languages


def test_faster_whisper_lang_classifier_detect(audio_data):
    classifier = FasterWhisperLangClassifier()
    language, probability = classifier.detect(audio_data.get_wav_data())
    assert isinstance(language, str)
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_faster_whisper_lang_classifier_audiochunk2array():
    audio_data = b"\x00\x01\x02\x03"
    array = FasterWhisperLangClassifier.audiochunk2array(audio_data)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32


def test_faster_whisper_stt_audiodata2array(audio_data):
    array = FasterWhisperSTT.audiodata2array(audio_data)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32


def test_faster_whisper_stt_invalid_model():
    stt = FasterWhisperSTT(config={"model": "invalid_model"})
    assert stt.config["model"] == "small"


def test_faster_whisper_lang_classifier_invalid_model():
    classifier = FasterWhisperLangClassifier(config={"model": "invalid_model"})
    assert classifier.config["model"] == "small"

if __name__ == "__main__":
    pytest.main()
