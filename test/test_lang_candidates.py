"""Auto-detect must choose from the configured languages, not from every one.

`execute` calls `detect_language(audio)` with no `valid_langs`, so the callee
falls back to `available_languages`, which is every key of
`FasterWhisperSTT.LANGUAGES`. A box configured for nl-NL with en-US secondary
then has its language picked from that whole set, and on real Dutch audio the
answer comes back as Latin, Finnish, Afrikaans, Danish or Russian depending on
the model.

These tests pin the call, not the model: they assert which candidate set
`execute` hands down, which is the part this plugin decides.
"""
from unittest.mock import MagicMock, patch

from ovos_stt_plugin_fasterwhisper import FasterWhisperSTT


def _stt_with_stub_engine():
    """An STT whose WhisperModel is stubbed, so no model is downloaded."""
    with patch("ovos_stt_plugin_fasterwhisper.WhisperModel", MagicMock()):
        stt = FasterWhisperSTT(config={"model": "tiny", "lang": "auto"})
    stt.engine = MagicMock()
    stt.engine.transcribe.return_value = ([], None)
    stt.engine.feature_extractor.sampling_rate = 16000
    return stt


def test_execute_restricts_detection_to_configured_languages():
    stt = _stt_with_stub_engine()
    audio = MagicMock()
    audio.get_np_float32.return_value = []

    with patch.object(FasterWhisperSTT, "detect_language",
                      return_value=("nl", 0.9)) as detect:
        with patch("ovos_stt_plugin_fasterwhisper.Configuration",
                   return_value={"lang": "nl-NL", "secondary_langs": ["en-US"]}):
            stt.lang = "auto"
            stt.execute(audio)

    assert detect.called, "execute did not run the auto-detect path"
    _args, kwargs = detect.call_args
    passed = kwargs.get("valid_langs")
    if passed is None and len(_args) > 1:
        passed = _args[1]
    assert passed is not None, (
        "execute called detect_language with no valid_langs, so the callee "
        "falls back to every language the model knows")
    assert {tag.split("-")[0].lower() for tag in passed} == {"nl", "en"}, (
        f"expected the configured languages, got {passed}")


def test_the_candidate_set_is_not_every_language():
    """The bug in one line: every known language as a candidate for a
    two-language box."""
    stt = _stt_with_stub_engine()
    audio = MagicMock()
    audio.get_np_float32.return_value = []

    with patch.object(FasterWhisperSTT, "detect_language",
                      return_value=("nl", 0.9)) as detect:
        with patch("ovos_stt_plugin_fasterwhisper.Configuration",
                   return_value={"lang": "nl-NL", "secondary_langs": ["en-US"]}):
            stt.lang = "auto"
            stt.execute(audio)

    _args, kwargs = detect.call_args
    passed = kwargs.get("valid_langs") or (_args[1] if len(_args) > 1 else None)
    assert passed is not None and len(passed) < len(FasterWhisperSTT.LANGUAGES)


def _candidates_for(config):
    """What execute hands down to detect_language under this configuration."""
    stt = _stt_with_stub_engine()
    audio = MagicMock()
    audio.get_np_float32.return_value = []
    with patch.object(FasterWhisperSTT, "detect_language",
                      return_value=("nl", 0.9)) as detect:
        with patch("ovos_stt_plugin_fasterwhisper.Configuration",
                   return_value=config):
            stt.lang = "auto"
            stt.execute(audio)
    args, kwargs = detect.call_args
    return kwargs.get("valid_langs") or (args[1] if len(args) > 1 else None)


def test_the_candidate_set_is_derived_from_the_configuration():
    """Changing the configured secondary changes what execute hands down.

    The set must be read from the configuration on each call, not fixed at
    construction and not taken from `self.lang`, which in this path is the
    literal "auto" that asked for detection.
    """
    nl_en = _candidates_for({"lang": "nl-NL", "secondary_langs": ["en-US"]})
    nl_de = _candidates_for({"lang": "nl-NL", "secondary_langs": ["de-DE"]})
    assert {t.split("-")[0].lower() for t in nl_en} == {"nl", "en"}
    assert {t.split("-")[0].lower() for t in nl_de} == {"nl", "de"}
    assert nl_en != nl_de, "the candidate set does not follow the configuration"


def test_one_configured_language_detects_only_that_one():
    """A box that speaks one language has one candidate, not the whole set."""
    for cfg in ({"lang": "nl-NL"},
                {"lang": "nl-NL", "secondary_langs": []},
                {"lang": "nl-NL", "secondary_langs": None}):
        got = _candidates_for(cfg)
        assert {t.split("-")[0].lower() for t in got} == {"nl"}, (
            f"{cfg} gave {got}")


def test_no_configured_language_still_restricts():
    """With nothing configured the set is still not every language."""
    got = _candidates_for({})
    assert got, "execute passed no candidate set at all"
    assert len(got) < len(FasterWhisperSTT.LANGUAGES)
