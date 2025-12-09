Unified summarization for YouTube, Web URLs, and PDF documents using modern retrieval + LLM pipelines.

Overview

OneView Summary provides a single interface to extract and summarize content from:

YouTube videos (via youtube-transcript-api)

Web pages (via WebBaseLoader)

PDF files (via pdfparser or any PDF text extraction backend)

It standardizes text extraction and feeds it into an LLM to produce concise summaries, bullet points, or Q/A outputs.

Features
YouTube

Fetch video transcripts using YouTubeTranscriptApi

Automatic language selection

Fallback to translation when available

Web URLs

Load and clean webpage content with WebBaseLoader

Handle multi-page content

Strip navigation, ads, and scripts

PDFs

Parse PDFs using pdfparser

Extract clean text even from multi-column documents

Optional chunking for long PDFs

Unified Workflow

All sources → cleaned text → LLM summary

Supports long-form summarization using chunking and recurrence

API and CLI modes available

Tech Stack
Component	Purpose
youtube-transcript-api	Transcript retrieval from YouTube videos
WebBaseLoader (LangChain)	Web page loading and extraction
pdfparser / PyPDF / PDFPlumber	PDF text extraction
Python 3.10+	Runtime
LangChain	Document loaders, chunking, and pipelines
LLM (OpenAI / Local / HF)	Summarization engine
Installation
git clone https://github.com/yourrepo/oneview-summary.git
cd oneview-summary

pip install -r requirements.txt


Typical dependencies:

youtube-transcript-api
langchain
langchain-community
pdfplumber
requests
openai
beautifulsoup4
tiktoken

Usage
1. Summarize a YouTube video
from youtube_transcript_api import YouTubeTranscriptApi
from oneview import summarize_text

video_id = "dQw4w9WgXcQ"
transcript = YouTubeTranscriptApi.get_transcript(video_id)
text = " ".join([t['text'] for t in transcript])

summary = summarize_text(text)
print(summary)

2. Summarize a Web URL
from langchain_community.document_loaders import WebBaseLoader
from oneview import summarize_text

url = "https://example.com/article"
docs = WebBaseLoader(url).load()
text = "\n".join([d.page_content for d in docs])

summary = summarize_text(text)
print(summary)

3. Summarize a PDF
import pdfplumber
from oneview import summarize_text

with pdfplumber.open("sample.pdf") as pdf:
    text = "\n".join([p.extract_text() or "" for p in pdf.pages])

summary = summarize_text(text)
print(summary)

Core Summarization Pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

def summarize_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    client = OpenAI()
    outputs = []
    for c in chunks:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": f"Summarize:\n{c}"}]
        )
        outputs.append(resp.choices[0].message["content"])

    return "\n\n".join(outputs)

Directory Structure
oneview-summary/
│
├── oneview/
│   ├── youtube.py
│   ├── web.py
│   ├── pdf.py
│   ├── summarize.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── examples/

Example CLI
python oneview.py --youtube https://youtu.be/dQw4w9WgXcQ
python oneview.py --url https://example.com/article
python oneview.py --pdf sample.pdf
