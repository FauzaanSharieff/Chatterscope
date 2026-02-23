import os
from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=key)

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input="Describe Marxism concisely in one sentence."
    )

    print("OpenAI says:", resp.output_text)


if __name__ == "__main__":
    main()