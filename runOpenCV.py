import os
import base64
import cv2

from dotenv import load_dotenv
from openai import OpenAI


def capture_frame(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Failed to capture image")

    return frame


def frame_to_base64(frame):
    ok, buf = cv2.imencode(".jpg", frame)

    if not ok:
        raise RuntimeError("Encoding failed")

    return base64.b64encode(buf).decode("utf-8")


def main():
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=key)

    print("Capturing image...")
    frame = capture_frame()

    img_b64 = frame_to_base64(frame)

    print("Sending to OpenAI...")

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe what you see in one short sentence."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img_b64}"
                }
            ]
        }]
    )

    print("Vision says:", resp.output_text.strip())


if __name__ == "__main__":
    main()