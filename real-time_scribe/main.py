import pyaudio
import asyncio
import json
import os
import sys
import websockets
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

startTime = datetime.now()

all_mic_data = []
all_transcripts = []

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8000

audio_queue = asyncio.Queue()

# SOAP instruction for the developer role
SOAP_INSTRUCTION = """
You are a clinical documentation assistant. Generate a SOAP note based on the following conversation between a clinician and a patient. Follow this strict structure:

- Subjective:
  - Chief Complaint (CC): [One sentence summarizing why the patient is seeking care.]
  - History of Present Illness (HPI): [A detailed paragraph summarizing the progression and current state of the patient's illness.]
  - History: [Past medical history, surgical history, medications, allergies, family history, social history — as much as available.]
  - Review of Systems (ROS): [Organized by system (General, Cardiovascular, Respiratory, GI, Musculoskeletal, Neurological, Psychiatric). If not mentioned, state "Not reported."]

- Objective:
  - General appearance.
  - Vital signs (if available; if not, state "Not available via video.")
  - Physical examination findings (if any).
  - Relevant lab or diagnostic information (if available).

- Assessment:
  - [List the major active medical problems and a brief status assessment.]

- Plan:
  - [Outline management plan including treatments, follow-ups, further discussions needed, etc.]

Tone: Formal, clinical, objective.
Format: Use bullet points inside each section when listing items; otherwise, use clear paragraphs.
Note: Fill in missing information with "Not specified" or "Not available" rather than making assumptions.
"""

# Used for microphone streaming only.
def mic_callback(input_data, frame_count, time_info, status_flag):
    audio_queue.put_nowait(input_data)
    return (input_data, pyaudio.paContinue)


def get_speaker_transcripts(json_data):
    speaker_transcripts = {}
    channel = json_data.get("channel", {})
    alternatives = channel.get("alternatives", [])

    for alternative in alternatives:
        for word_info in alternative.get("words", []):
            speaker_id = word_info.get("speaker", "Unknown")
            punctuated_word = word_info.get("punctuated_word", word_info.get("word", ""))
            if speaker_id not in speaker_transcripts:
                speaker_transcripts[speaker_id] = []
            speaker_transcripts[speaker_id].append(punctuated_word)

    # Format the output
    formatted_transcripts = []
    for speaker_id, words in speaker_transcripts.items():
        formatted_transcripts.append(f"Speaker {speaker_id}: {', '.join(words)}")

    return "\n".join(formatted_transcripts)


async def sender(ws, audio_queue):
    print("🟢 Ready to stream audio to Deepgram")
    try:
        while True:
            mic_data = await audio_queue.get()
            all_mic_data.append(mic_data)
            await ws.send(mic_data)
    except asyncio.CancelledError:
        return
    except websockets.exceptions.ConnectionClosedOK:
        await ws.send(json.dumps({"type": "CloseStream"}))
        print("🟢 Successfully closed Deepgram connection")
    except Exception as e:
        print(f"Error while sending: {str(e)}")
        raise


async def receiver(ws):
    first_message = True
    transcript = ""

    async for msg in ws:
        res = json.loads(msg)
        if first_message:
            print("🟢 Successfully receiving Deepgram messages")
            first_message = False
        try:
            if res.get("is_final"):
                transcript = get_speaker_transcripts(res)
                if transcript:
                    print(transcript)
                    all_transcripts.append(transcript)
        except KeyError:
            print(f"🔴 ERROR: Received unexpected API response! {msg}")


async def microphone(audio_queue):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=mic_callback,
    )

    stream.start_stream()

    global SAMPLE_SIZE
    SAMPLE_SIZE = audio.get_sample_size(FORMAT)

    print("🎙️  Microphone started. Press Ctrl+C once to stop.")
    try:
        while stream.is_active():
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


async def run(key, **kwargs):
    deepgram_url = f'wss://api.deepgram.com/v1/listen?punctuate=true&diarize=true'
    deepgram_url += f"&model={kwargs['model']}"
    deepgram_url += "&encoding=linear16&sample_rate=16000"

    async with websockets.connect(
        deepgram_url, extra_headers={"Authorization": "Token {}".format(key)}
    ) as ws:
        print("🟢 Successfully opened Deepgram streaming connection")

        # Set up tasks
        tasks = [
            asyncio.ensure_future(sender(ws, audio_queue)),
            asyncio.ensure_future(receiver(ws)),
            asyncio.ensure_future(microphone(audio_queue)),
        ]

        # Wait for tasks, allow Ctrl+C to cancel
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # Quietly exit when tasks are cancelled
            generate_note_and_save(all_transcripts, "generated_soap_note.txt")


def generate_note_and_save(all_transcripts, output_file):
    # Combine all transcripts into a single string
    combined_transcript = "\n".join(all_transcripts)

    # Initialize OpenAI client
    client = OpenAI()

    # Generate SOAP note using OpenAI
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "developer", "content": SOAP_INSTRUCTION},
            {"role": "user", "content": combined_transcript},
        ],
    )

    # Save the generated SOAP note to a file
    with open(output_file, "w") as file:
        file.write(response.output_text)

    print(f"📝 SOAP note saved to {output_file}")


def main():
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        print("Please set the DEEPGRAM_API_KEY environment variable.")
        sys.exit(1)
    format = "text"
    model = 'nova-3-medical'

    asyncio.run(run(key, model=model))


if __name__ == "__main__":
    main()