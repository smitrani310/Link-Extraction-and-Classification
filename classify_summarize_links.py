import os
import uuid
import argparse
import logging
import subprocess
import requests
from typing import List, Optional

import pandas as pd
import whisper
import openai
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime

# ------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------
# Create a "Logs" folder if it doesn't exist
log_folder = "Logs"
os.makedirs(log_folder, exist_ok=True)

# Generate a timestamped log filename
timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_filename = os.path.join(log_folder, f"{timestamp_str}.log")

# Create a logger and set it to the lowest level we plan to capture
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # We want to capture everything; we can filter later per handler

# Create a file handler to write all logs (DEBUG and above)
file_handler = logging.FileHandler(log_filename, mode="w")
file_handler.setLevel(logging.DEBUG)

# Create a console handler to show only Warnings and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Define a log format
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# Attach the format to both handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Test log messages
logger.debug("This is a DEBUG message - will only appear in the file.")
logger.info("This is an INFO message - will only appear in the file.")
logger.warning("This is a WARNING message - appears in console AND file.")
logger.error("This is an ERROR message - appears in console AND file.")
logger.critical("This is a CRITICAL message - appears in console AND file.")

# ------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------
env_path = r"./Environment_Variables.env"
load_dotenv(env_path)

openai_api_key = os.getenv("OPENAI_API_KEY", "")

# ------------------------------------------------------------------
# OOP Classes
# ------------------------------------------------------------------

class VideoDownloader:
    """
    Responsible for downloading audio from a video link using yt-dlp.
    """
    def __init__(self, output_folder: str = "downloads"):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def download_audio(self, url: str) -> str:
        """
        Downloads the audio track from a given video URL and saves it as an MP3 file.
        Returns the path to the downloaded audio file.
        Raises subprocess.CalledProcessError if the download fails.
        """
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_filepath = os.path.join(self.output_folder, audio_filename)

        command = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--output", audio_filepath,
            url
        ]

        logger.info(f"Attempting to download audio from {url}...")
        subprocess.run(command, check=True)  # Raises CalledProcessError if fail
        logger.info(f"Audio downloaded to {audio_filepath}")
        return audio_filepath


class Transcriber:
    """
    Handles loading the Whisper model and transcribing audio.
    """
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        :param model_name: Whisper model name (e.g., 'base', 'small', 'medium', 'large').
        :param device: 'cuda' for GPU, 'cpu' for CPU, or None to auto-detect.
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Whisper model: {self.model_name} on device: {self.device}")
        self.model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the given audio file using the loaded Whisper model.
        Returns the transcript as a string.
        """
        logger.info(f"Transcribing audio: {audio_path}")
        result = self.model.transcribe(audio_path)
        transcript = result.get("text", "")
        logger.debug(f"Transcription result (first 100 chars): {transcript[:100]}...")
        return transcript


class TextClassifier:
    """
    Uses GPT to classify or summarize text.
    """
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", max_retries: int = 3):
        """
        :param api_key: OpenAI API key
        :param model_name: e.g., 'gpt-3.5-turbo' or 'gpt-4'
        :param max_retries: number of times to retry GPT on failure
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model_name = model_name
        self.max_retries = max_retries

    def classify_text(self, text: str, categories: List[str]) -> str:
        """
        Sends text to GPT for classification into specified categories (1 or 2).
        Expected format: 'TAGS / summary'.
        """
        if not self.api_key:
            logger.error("No OpenAI API key provided. Classification may fail.")
        categories_str = ", ".join(categories)

        system_prompt = (
            f"You are a text classifier. You will be given text, and you must classify it into "
            f"exactly ONE or TWO of these categories: {categories_str}. "
            f"Reply category name/s. Then provide a single-line summary. "
            f"Format: 'TAGS / Summarize'."
        )

        user_prompt = f"Text to classify: \n{text}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=50
                )
                classification = response.choices[0].message.content.strip()
                logger.debug(f"GPT classification (attempt {attempt}): {classification}")
                return classification
            except Exception as e:
                logger.error(f"GPT classification failed (attempt {attempt}). Error: {e}")
                if attempt == self.max_retries:
                    return "NOT_CLASSIFIED / Failed to classify."

        return "NOT_CLASSIFIED / Unexpected error."

    def summarize_text(self, text: str) -> str:
        """
        A simple GPT-based text summarizer returning a single-line summary.
        Format: 'WEBSITE_SUMMARY / <summary>'.
        """
        if not self.api_key:
            logger.error("No OpenAI API key provided. Summarization may fail.")

        system_prompt = (
            "You are a helpful assistant that summarizes webpage content. "
            "Output only a single line summary of the text. "
            "Format: 'WEBSITE_SUMMARY / <one-line summary>'."
        )
        user_prompt = f"Text to summarize:\n{text}"

        for attempt in range(1, self.max_retries + 1):
            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=60
                )
                gpt_result = response.choices[0].message.content.strip()
                logger.debug(f"GPT summary (attempt {attempt}): {gpt_result}")
                return gpt_result
            except Exception as e:
                logger.error(f"GPT summarization failed (attempt {attempt}). Error: {e}")
                if attempt == self.max_retries:
                    return "WEBSITE_SUMMARY / Failed to summarize."

        return "WEBSITE_SUMMARY / Unexpected error."


class WebsiteSummarizer:
    """
    Fetches webpage HTML, parses out main text, and uses GPT to summarize it.
    """
    def __init__(self, text_classifier: TextClassifier):
        self.text_classifier = text_classifier

    def fetch_and_summarize(self, url: str) -> str:
        """
        Fetches the webpage at `url`, extracts readable text, and asks GPT to summarize it.
        Returns a string: 'WEBSITE_SUMMARY / <summary>'.
        """
        try:
            logger.info(f"Fetching webpage content from {url}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad status
        except Exception as e:
            err_msg = f"WEBSITE_SUMMARY / Request error: {e}"
            logger.error(err_msg)
            return err_msg

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # A very naive approach: gather all text from <p> tags (you can refine this a lot)
        paragraphs = soup.find_all("p")
        text_content = "\n".join(p.get_text() for p in paragraphs)

        # If there's no <p> text, fallback to the entire HTML's .get_text()
        if not text_content.strip():
            text_content = soup.get_text()

        # Summarize with GPT
        summary_result = self.text_classifier.summarize_text(text_content)
        return summary_result


class VideoClassificationPipeline:
    """
    Coordinates the downloading/transcribing/classifying steps for video,
    and falls back to website summarization if it's not a video link.
    """
    def __init__(
            self,
            downloader: VideoDownloader,
            transcriber: Transcriber,
            classifier: TextClassifier,
            website_summarizer: WebsiteSummarizer,
            default_categories: Optional[List[str]] = None
    ):
        self.downloader = downloader
        self.transcriber = transcriber
        self.classifier = classifier
        self.website_summarizer = website_summarizer
        self.default_categories = default_categories or [
            "AI", "Cooking", "Travel", "Entertainment", "Marketing", "Tutorial"
        ]

    def is_video_domain(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower()
        # Add whichever video-hosting domains you support:
        known_video_domains = [
            "youtube.com", "youtu.be",
            "vimeo.com", "facebook.com",
            "tiktok.com", "instagram.com",
            "dailymotion.com", "twitch.tv"
        ]
        return any(d in domain for d in known_video_domains)

    def classify_video_or_website(self, url: str, categories: Optional[List[str]] = None) -> str:
        cats = categories or self.default_categories

        # 1. Check if it's a recognized video domain first
        if self.is_video_domain(url):
            # Attempt video download
            try:
                audio_file = self.downloader.download_audio(url)
            except subprocess.CalledProcessError as e:
                # Fallback to website summarization
                logger.warning(f"Download failed; falling back to website summarization for {url}. Error: {e}")
                return self.website_summarizer.fetch_and_summarize(url)
            except Exception as e:
                msg = f"NOT_VIDEO / Download error: {e}"
                logger.error(msg)
                return msg

            # 2. Transcribe
            try:
                transcript = self.transcriber.transcribe(audio_file)
            except Exception as e:
                msg = f"NOT_TRANSCRIBED / Transcription error: {e}"
                logger.error(msg)
                return msg

            # 3. Classify
            classification_result = self.classifier.classify_text(transcript, cats)
            return classification_result
        else:
            # If it's NOT a known video domain, just treat it as a website
            return self.website_summarizer.fetch_and_summarize(url)


# ------------------------------------------------------------------
# Main function for CLI usage
# ------------------------------------------------------------------
def main(custom_categories: List[str]):
    parser = argparse.ArgumentParser(description="Process an Excel or CSV file of URLs.")
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the input file (Excel or CSV) containing URLs."
    )
    parser.add_argument(
        "--link-column",
        type=str,
        default="Links",
        help="Name of the column in the file that contains the URLs."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.xlsx",
        help="Output file path for the updated data (default: output.xlsx)."
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Name of the GPT model to use (e.g., 'gpt-3.5-turbo', 'gpt-4')."
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="small",
        help="Local Whisper model name (e.g., 'base', 'small', 'medium', 'large')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for Whisper ('cuda' or 'cpu'). If None, auto-detect."
    )
    args = parser.parse_args()
    input_file = args.filepath
    link_col = args.link_column
    output_file = args.output

    # -------------------------------------------
    # 1. Check if input file exists
    # -------------------------------------------
    if not os.path.exists(input_file):
        logger.error(f"The file '{input_file}' does not exist.")
        return

    # -------------------------------------------
    # 2. Determine file type and read data
    # -------------------------------------------
    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_file)
    elif ext in [".xls", ".xlsx", ".xlsm"]:
        df = pd.read_excel(input_file)
    else:
        logger.error(f"Unsupported file extension '{ext}'. Please provide CSV or Excel.")
        return

    # -------------------------------------------
    # 3. Verify link column
    # -------------------------------------------
    if link_col not in df.columns:
        logger.error(f"Column '{link_col}' not found in the file.")
        logger.info(f"Available columns: {list(df.columns)}")
        return

    # -------------------------------------------
    # 4. Initialize pipeline components
    # -------------------------------------------
    downloader = VideoDownloader(output_folder="downloads")
    transcriber = Transcriber(model_name=args.whisper_model, device=args.device)
    classifier = TextClassifier(api_key=openai_api_key, model_name=args.gpt_model, max_retries=3)
    web_summarizer = WebsiteSummarizer(text_classifier=classifier)

    pipeline = VideoClassificationPipeline(
        downloader=downloader,
        transcriber=transcriber,
        classifier=classifier,
        website_summarizer=web_summarizer,
        default_categories=custom_categories
    )

    # -------------------------------------------
    # 5. Iterate over links with a progress bar
    # -------------------------------------------
    logger.info("Processing URLs...")
    df["TAGS"] = ""
    df["Summary"] = ""
    df["Error"] = ""  # Keep track of errors if any

    links = df[link_col].tolist()
    for i in tqdm(range(len(links)), desc="Processing links"):
        url = links[i]
        result = pipeline.classify_video_or_website(url, categories=custom_categories)

        # Expecting either "TAGS / Summarize" or "WEBSITE_SUMMARY / Summarize"
        # or "NOT_VIDEO / some error", etc.
        if "/" in result:
            parts = result.split("/", 1)
            tag_str = parts[0].strip()
            summary_str = parts[1].strip() if len(parts) > 1 else ""

            # If the tag is something like "WEBSITE_SUMMARY", store that in TAGS, the rest in Summary
            # Or if it's "NOT_VIDEO", store in Error, etc.
            if tag_str.startswith("NOT_") or tag_str.startswith("WEBSITE_SUMMARY"):
                # We'll handle those as special cases
                if tag_str.startswith("NOT_"):
                    df.at[i, "Error"] = result
                else:
                    # It's a normal fallback for a website
                    df.at[i, "TAGS"] = tag_str
                    df.at[i, "Summary"] = summary_str
            else:
                # Normal case: "TAGS / summary"
                df.at[i, "TAGS"] = tag_str
                df.at[i, "Summary"] = summary_str
        else:
            # Unexpected format
            df.at[i, "Error"] = result

    # -------------------------------------------
    # 6. Save the resulting DataFrame
    # -------------------------------------------
    df.to_excel(output_file, index=False)
    logger.info(f"Done! Results saved to {output_file}")


if __name__ == "__main__":
    # Define default categories for video classification
    custom_categories = [
        "AI", "AI Tools", "Job Search", "Fitness", "Productivity",
        "Tutorial", "Music", "Comedy", "Other"
    ]
    main(custom_categories)
