import click
import os
from speech_recognition import Recognizer, AudioFile

from ovos_stt_plugin_fasterwhisper import FasterWhisperSTT


@click.command()
@click.option("--path")
@click.option("--lang", default="en-us")
@click.option("--model", default="base")
@click.option("--format", default="wav")
@click.option("--beam", default=5)
@click.option("--cuda", default=False)
@click.option("--compute", default="int8")
@click.option("--vad", default=False)
def transcribe(path: str, lang: str, model: str, format: str, beam: int, cuda: bool, compute: str, vad: bool):
    config = {
        "lang": lang,
        "model": model,
        "beam_size": beam,
        "use_cuda": cuda,
        "compute_type": compute,
        "vad_filter": vad
    }

    b = FasterWhisperSTT(config=config)

    if os.path.isfile(path):
        try:
            with AudioFile(path) as source:
                try:
                    audio = Recognizer().record(source)
                    t = b.execute(audio, language="en")
                    print(t)
                    with open(path.replace(f'.{format}', '.txt'), "w") as f:
                        f.write(t)
                except:
                    print("failed to transcribe file")
        except:
            print("failed to open file", f)
    elif os.path.isdir(path):
        for root, folder, files in os.walk(path):
            for f in files:
                if f.endswith(".wav") and not os.path.isfile(f"{root}/{f.replace(f'.{format}', '.txt')}"):
                    print(root, f)
                    try:
                        with AudioFile(f"{root}/{f}") as source:
                            try:
                                audio = Recognizer().record(source)
                                t = b.execute(audio, language="en")
                                print(t)
                                with open(f"{root}/{f.replace(f'.{format}', '.txt')}", "w") as f:
                                    f.write(t)
                            except:
                                print("failed to transcribe file")
                    except:
                        print("failed to open file", f)
                        continue


if __name__ == "__main__":
    transcribe()
