import os
import base64
import cv2
import winsound

from dotenv import load_dotenv
from openai import OpenAI
from google.cloud import texttospeech


def capture_frame(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera_index=1 or 2.")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to capture frame.")
    return frame


def frame_to_base64_jpeg(frame, jpeg_quality=85):
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def describe_scene(openai_client: OpenAI, image_b64: str) -> str:
    resp = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe what you see in one short sentence. Focus on main objects. In the second sentence, follow up with a witty pun."},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ],
        }],
    )
    return resp.output_text.strip()


def speak_text_google(text: str, out_file="speech.wav"):
    client = texttospeech.TextToSpeechClient()

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="en-GB",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        ),
    )

    with open(out_file, "wb") as f:
        f.write(response.audio_content)

    winsound.PlaySound(out_file, winsound.SND_FILENAME)


def main():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    openai_client = OpenAI(api_key=openai_key)

    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not google_creds:
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS in .env")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds

    print("Capturing image...")
    frame = capture_frame(camera_index=0)

    print("Encoding image...")
    img_b64 = frame_to_base64_jpeg(frame)

    print("Thinking...")
    description = describe_scene(openai_client, img_b64)
    print("Subtitles:", description)

    print("Chattering...")
    speak_text_google(description)


if __name__ == "__main__":
    main()