from openai import OpenAI
import sys
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

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

def transcribe_audio(audio_file):
    # STEP 1: Create a Deepgram client using the API key
    deepgram = DeepgramClient()

    with open(audio_file, "rb") as file:
        buffer_data = file.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    # STEP 2: Configure Deepgram options for audio analysis
    options = PrerecordedOptions(
        model="nova-3-medical",
        smart_format=True,
        diarize=True
    )

    # STEP 3: Call the transcribe_file method with the text payload and options
    response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

    # STEP 4: Extract and return the transcript
    data = response["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"]

    return data

def generate_note_and_save(transcript, output_file):
    # Call the OpenAI API to generate the SOAP note
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "developer",
                "content": SOAP_INSTRUCTION
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    )

    # Save the generated SOAP note to a file
    with open(output_file, "w") as file:
        file.write(response.output_text)

    print(f"SOAP note saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_file = "generated_soap_note.txt"

    print("Starting transcription...")
    transcript = transcribe_audio(audio_file)
    print("Transcription completed.")

    print("Generating SOAP note...")
    generate_note_and_save(transcript, output_file)
    print("SOAP note generation completed.")

if __name__ == "__main__":
    main()