import os
import winsound

from dotenv import load_dotenv
from google.cloud import texttospeech


def main():
    load_dotenv()

    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS in .env")

    # Tell the Google library where the credentials are
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

    client = texttospeech.TextToSpeechClient()

    text = input("Type something for me to say: ").strip()
    if not text:
        print("No text entered. Exiting.")
        return

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="de-DE",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        ),
    )

    out_file = "speech.wav"
    with open(out_file, "wb") as f:
        f.write(response.audio_content)

    print("Playing audio...")
    winsound.PlaySound(out_file, winsound.SND_FILENAME)


if __name__ == "__main__":
    main()