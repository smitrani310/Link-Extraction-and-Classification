import os
import sys
import uuid
import argparse
import subprocess
import openai
import whisper
import pandas as pd
from dotenv import load_dotenv

# ------------------------------------------------------------------------
# Load environment variables (.env must define OPENAI_API_KEY, if not in env)
# ------------------------------------------------------------------------
env_path = r"./Environment_Variables.env"
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"}
AUDIO_OUTPUT_FOLDER = "extracted_audio"

# ------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------


def extract_audio(video_path: str, output_folder: str = AUDIO_OUTPUT_FOLDER) -> str:
    """
    Extracts audio from the given video file using ffmpeg and saves it as an MP3.
    Returns the path to the extracted audio file, or empty string on error.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique MP3 file name
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_filepath = os.path.join(output_folder, audio_filename)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "mp3",
        "-y",
        audio_filepath
    ]

    try:
        # Capture output to avoid spam; check=True raises CalledProcessError on fail
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed extracting audio from {video_path}: {e}")
        return ""

    return audio_filepath


def transcribe_audio_whisper(audio_path: str, whisper_model) -> str:
    """
    Transcribes the audio file using a *preloaded* Whisper model.
    Returns the transcription text, or empty string if fails.
    """
    if not os.path.exists(audio_path):
        return ""
    try:
        result = whisper_model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        print(f"[ERROR] Transcription failed for {audio_path}: {e}")
        return ""


def classify_and_summarize_gpt(text: str, gpt_model: str) -> (str, str, str):
    """
    Sends the transcript to OpenAI GPT for classification into TWO tags + one-line summary.
    Returns (tag1, tag2, summary).
    """
    if not openai.api_key:
        print("[WARNING] No OPENAI_API_KEY found. Classification may fail.")
        return ("NOT_CLASSIFIED", "", "No API key")

    system_prompt = (
        "You are a helpful assistant. You will receive text (a transcript), and you must:\n"
        "1) Provide exactly TWO relevant tags or categories (or one if only one truly applies).\n"
        "2) Provide a single-line summary.\n\n"
        "Format your response as: 'Tag1, Tag2 / <summary>'\n\n"
        "Example: 'AI, Marketing / This video discusses how AI impacts marketing strategies.'"
    )

    user_prompt = f"Transcript:\n{text}"

    try:
        response = openai.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=60
        )
        classification = response.choices[0].message.content.strip()

        # Expect "Tag1, Tag2 / summary" or "Tag1 / summary"
        if "/" not in classification:
            # If GPT doesn't follow format, handle gracefully
            return ("NOT_FORMATTED", "", classification[:100])

        parts = classification.split("/", 1)
        tags_str = parts[0].strip()   # e.g. "AI, Marketing"
        summary_str = parts[1].strip()

        # Split tags on comma
        if "," in tags_str:
            tag_parts = tags_str.split(",", 1)
            tag1 = tag_parts[0].strip()
            tag2 = tag_parts[1].strip()
        else:
            tag1 = tags_str
            tag2 = ""

        return (tag1, tag2, summary_str)

    except Exception as e:
        print(f"[ERROR] GPT classification failed: {e}")
        return ("NOT_CLASSIFIED", "", "Summary not available")


def main():
    # --------------------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Process video files: extract audio, transcribe, classify, summarize.")
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the root folder containing video files (recursively scanned)."
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-3.5-turbo",
        help="GPT model name (e.g. 'gpt-3.5-turbo' or 'gpt-4')."
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        help="Local Whisper model name (e.g., 'base', 'small', 'medium', 'large')."
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=["csv", "excel"],
        help="Choose output format: 'csv' or 'excel' (default: csv)."
    )

    # ---------------------------------------------------------
    # Dynamically build the default output filename
    # based on the input_folder name.
    # ---------------------------------------------------------
    args_tmp, _ = parser.parse_known_args()  # partial parse
    input_folder_tmp = args_tmp.input_folder
    folder_name = os.path.basename(os.path.normpath(input_folder_tmp))
    default_outfile = f"{folder_name}_summarized"

    parser.add_argument(
        "--output-file",
        type=str,
        default=default_outfile,
        help="Base filename (without extension). "
             f"Default is '<foldername>_summarized'."
    )

    args = parser.parse_args()
    input_folder = args.input_folder
    gpt_model = args.gpt_model
    whisper_model_name = args.whisper_model
    output_format = args.output_format
    output_file_basename = args.output_file  # We'll append .csv or .xlsx

    # --------------------------------------------------------------------
    # 1) Load Whisper model once outside the loop
    # --------------------------------------------------------------------
    print(f"[INFO] Loading Whisper model: {whisper_model_name}")
    whisper_model = whisper.load_model(whisper_model_name)

    # --------------------------------------------------------------------
    # 2) Prepare a list to hold results
    # --------------------------------------------------------------------
    results = []

    # --------------------------------------------------------------------
    # 3) Walk through the input folder to find video files
    # --------------------------------------------------------------------
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in VIDEO_EXTENSIONS:
                video_path = os.path.join(dirpath, filename)
                print(f"[INFO] Found video: {video_path}")

                # Extract audio
                audio_path = extract_audio(video_path)
                if not audio_path:
                    print("[ERROR] Audio extraction failed, skipping classification.")
                    results.append({
                        "File Name": filename,
                        "Tag1": "EXTRACTION_FAILED",
                        "Tag2": "",
                        "Summary": "Audio extraction error",
                        "Path": video_path
                    })
                    continue

                # Transcribe
                transcript = transcribe_audio_whisper(audio_path, whisper_model)
                if not transcript:
                    print("[ERROR] Transcription failed or empty.")
                    results.append({
                        "File Name": filename,
                        "Tag1": "TRANSCRIPTION_FAILED",
                        "Tag2": "",
                        "Summary": "No transcript",
                        "Path": video_path
                    })
                    continue

                # Classify & Summarize
                tag1, tag2, summary = classify_and_summarize_gpt(transcript, gpt_model)

                # Add to results
                results.append({
                    "File Name": filename,
                    "Tag1": tag1,
                    "Tag2": tag2,
                    "Summary": summary,
                    "Path": video_path
                })

    # --------------------------------------------------------------------
    # 4) Convert results to DataFrame
    # --------------------------------------------------------------------
    df = pd.DataFrame(results, columns=["File Name", "Tag1", "Tag2", "Summary", "Path"])

    # --------------------------------------------------------------------
    # 5) Output based on chosen format
    # --------------------------------------------------------------------
    if output_format == "csv":
        out_file = f"{output_file_basename}.csv"
        df.to_csv(out_file, index=False, encoding="utf-8")
        print(f"[INFO] Saved CSV to {out_file}")

    else:  # excel
        out_file = f"{output_file_basename}.xlsx"
        df.to_excel(out_file, index=False)
        print(f"[INFO] Saved Excel to {out_file}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage example:\n"
              "  python video_audio_classify.py /path/to/videos "
              "--gpt-model gpt-4o --whisper-model medium --output-format excel --output-file my_videos")
    main()
