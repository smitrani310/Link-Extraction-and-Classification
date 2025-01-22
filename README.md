# Link Extraction and Classification Project

This project contains two main scripts:

1. **`extract_links.py`**  
   - Reads a text document and extracts all URLs (links) found in that file.

2. **`classify_summarize_links.py`**  
   - Reads an Excel file containing video or webpage links, downloads/transcribes (if video), classifies the content using OpenAI’s GPT models, or summarizes webpages.  
   - Outputs the classification and summary to a new Excel file.

---

## Table of Contents

1. [Features](#features)  
2. [Installation & Requirements](#installation--requirements)  
3. [Usage](#usage)  
   - [1) Extracting Links from a Text Document](#1-extracting-links-from-a-text-document)  
   - [2) Classifying and Summarizing Links from Excel](#2-classifying-and-summarizing-links-from-excel)  
4. [Environment Variables](#environment-variables)  
5. [Logging](#logging)  
6. [Known Issues & Limitations](#known-issues--limitations)  
7. [License](#license)

---

## Features

- **Link Extraction**: Quickly parse a `.txt` file and retrieve all valid URLs into a separate list or file.  
- **Video & Webpage Handling**:  
  - **Video**: Automatically download the audio with [yt-dlp](https://github.com/yt-dlp/yt-dlp), transcribe with [OpenAI Whisper](https://github.com/openai/whisper), then classify & summarize with GPT.  
  - **Webpage**: If the URL is **not** a recognized video domain, fetch the webpage HTML, extract main text, and summarize it with GPT.  
- **Configurable & Extensible**:  
  - Choose different Whisper and GPT model names.  
  - Pass your own custom categories for classification.  
- **Logging**:  
  - Both scripts log relevant events.  
  - A log file is created in the `Logs/` folder each time you run the scripts, named with a timestamp.

---

## Installation & Requirements

1. **Python 3.8+** recommended.  
2. Install dependencies from `requirements.txt` or manually:
   pip install -r requirements.txt

---

## Usage
## Extracting Links from a Text Document

 - **Script**: extract_links.py

 - **Purpose**: Reads a .txt file, finds all URLs, and writes them to an output file or prints them on the console.

 - Create or place your text file (e.g., sample_input.txt) with URLs.
 Run:
 'python extract_links.py --input sample_input.txt --output extracted_links.txt'

- *input:* path to the text file containing potential links.
- *output:* path to save the extracted links (if not specified, it might print to console or use a default file).

*Result:*

A new file extracted_links.txt containing one URL per line.
Logs will be printed to console and also saved in Logs/<timestamp>.log.

---

## Classifying and Summarizing Links from Excel
- **Script**: classify_summarize_links.py

- **Purpose**: Iterates over an Excel file containing URLs, attempts to:

 - Download & transcribe if it’s a video (YouTube, Vimeo, etc.).
 - Summarize if it’s a website.
 - Classify text output via GPT into configured categories.
 - Save results (including tags, summaries, and error messages) in a new Excel file.
 
 - (1). Prepare your Excel or CSV file with a column (e.g. "Links") containing URLs.
 - (2). python classify_summarize_links.py my_links.xlsx --link-column "Links" --output "classified_links.xlsx" \
    --gpt-model "gpt-3.5-turbo" \
    --whisper-model "small" \
    --device "cuda"

  - *my_links.xlsx*: your input file.
  - *link-column:* column name in the Excel/CSV that contains the links.
  - *output:* name/path of the output Excel file (default is output.xlsx).
  - *pt-model:* which OpenAI model to use (e.g., gpt-3.5-turbo, gpt-4).
  - *whisper-model:* which Whisper model to use (e.g., base, small, medium, large).
  - *device:* cuda (GPU) or cpu. If omitted, the script auto-detects GPU availability.

*Result:*

An Excel file (e.g. classified_links.xlsx) with added columns like:
 - *TAGS:* The classification categories or WEBSITE_SUMMARY.
 - *Summary:* The summarized text.
 - *Error:* Any error messages (e.g., failed downloads, classification issues).
 - Logs printed to console, with a more detailed log file in Logs/<timestamp>.log.

---

## Environment Variables
Both scripts may rely on environment variables for API Keys or other secrets. 
Create .env file named : "Environment_Variables.env" with:
 - OPENAI_API_KEY="YourOpenAIKeyHere"
 
Ensure the .env file is referenced by your scripts:
	'from dotenv import load_dotenv'
	'load_dotenv("path/to/Environment_Variables.env")'
This way, openai.api_key can be set via os.getenv("OPENAI_API_KEY").

---

## Logging
Each script configures Python logging to:

Print “relevant” logs (e.g., warnings, errors) to the console.
Write all logs (e.g., INFO, DEBUG) to a timestamped log file in Logs/<YYYY_MM_DD_HH_MM_SS>.log.
You can inspect the log files if you need to debug issues (e.g., network failures, transcription errors).

---

## Known Issues & Limitations
 - *Token Limits:* If summarizing very large webpages, GPT might exceed token limits. You may need to split text into smaller chunks or use an approach that handles more content.
 - *Video Domain Identification:* A simple domain check is used for recognized video sites. If the site isn’t recognized, it’s treated as a website.
 - *Parallelization:* The scripts process links sequentially. For large datasets, consider multiprocessing or asynchronous logic.

*Dependencies:*
 - yt-dlp requires ffmpeg.
 - whisper + torch can be resource-intensive.
 - OpenAI GPT API calls can fail if rate-limited or if the API key is invalid.


---

## License

> Distributed under the [Apache 2.0](LICENSE).  
> **© 2024 Smitrani310@gmail.com. All rights reserved.**


---
 
