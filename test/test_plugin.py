import numpy as np
import pytest
from speech_recognition import AudioFile, Recognizer

from ovos_plugin_manager.utils.audio import AudioData

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


def test_audiochunk2array():
    # raw PCM chunk: 2 int16 samples (sample_width=2)
    chunk = b"\x00\x01\x02\x03"
    array = AudioData(chunk, sample_rate=16000, sample_width=2).get_np_float32()
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert len(array) == 2
    # values normalised to [-1.0, 1.0]
    assert np.all(np.abs(array) <= 1.0)


def test_audiodata2array(audio_data):
    array = audio_data.get_np_float32()
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert len(array) > 0
    assert np.all(np.abs(array) <= 1.0)


if __name__ == "__main__":
    pytest.main()
