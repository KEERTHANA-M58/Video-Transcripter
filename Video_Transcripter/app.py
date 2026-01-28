import os
import importlib

# Dynamically import optional dependencies to avoid static "import could not be resolved" problems
def _try_import(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None

_streamlit = _try_import("streamlit")
st = getattr(_streamlit, "st", _streamlit) if _streamlit is not None else None

_langchain_embeddings = _try_import("langchain_community.embeddings")
OpenAIEmbeddings = getattr(_langchain_embeddings, "OpenAIEmbeddings", None) if _langchain_embeddings is not None else None

_langchain_vectorstores = _try_import("langchain_community.vectorstores")
FAISS = getattr(_langchain_vectorstores, "FAISS", None) if _langchain_vectorstores is not None else None

_langchain_loaders = _try_import("langchain_community.document_loaders")
TextLoader = getattr(_langchain_loaders, "TextLoader", None) if _langchain_loaders is not None else None

_text_splitter = _try_import("langchain.text_splitter")
CharacterTextSplitter = getattr(_text_splitter, "CharacterTextSplitter", None) if _text_splitter is not None else None

_chains = _try_import("langchain.chains")
RetrievalQA = getattr(_chains, "RetrievalQA", None) if _chains is not None else None

_llms = _try_import("langchain_community.llms")
OpenAI = getattr(_llms, "OpenAI", None) if _llms is not None else None

_pytube = _try_import("pytube")
YouTube = getattr(_pytube, "YouTube", None) if _pytube is not None else None

_ytt = _try_import("youtube_transcript_api")
YouTubeTranscriptApi = getattr(_ytt, "YouTubeTranscriptApi", None) if _ytt is not None else None
TranscriptsDisabled = getattr(_ytt, "TranscriptsDisabled", Exception) if _ytt is not None else Exception
NoTranscriptFound = getattr(_ytt, "NoTranscriptFound", Exception) if _ytt is not None else Exception
VideoUnavailable = getattr(_ytt, "VideoUnavailable", Exception) if _ytt is not None else Exception
CouldNotRetrieveTranscript = getattr(_ytt, "CouldNotRetrieveTranscript", Exception) if _ytt is not None else Exception

_dotenv = _try_import("dotenv")
load_dotenv = getattr(_dotenv, "load_dotenv", lambda: None)
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        # Prefer get_transcript which returns a list of {"text","start","duration"}
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item.get("text", "") for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""


def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


st.title("AI-Powered Tutor")
st.write("Ask questions from YouTube lecture transcripts.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("Transcript processed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.write("**Answer:**", answer)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        # Prefer get_transcript which returns a list of {"text","start","duration"}
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item.get("text", "") for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""


def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


st.title("AI-Powered Tutor")
st.write("Ask questions from YouTube lecture transcripts.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("Transcript processed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.write("**Answer:**", answer)
import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        # Prefer get_transcript which returns a list of {"text","start","duration"}
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item.get("text", "") for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""


def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


st.title("AI-Powered Tutor")
st.write("Ask questions from YouTube lecture transcripts.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("Transcript processed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.write("**Answer:**", answer)
import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        # Prefer get_transcript which returns a list of {"text","start","duration"}
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item.get("text", "") for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""


def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


st.title("AI-Powered Tutor")
st.write("Ask questions from YouTube lecture transcripts.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("Transcript processed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.write("**Answer:**", answer)
import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        # Prefer get_transcript which returns a list of {"text","start","duration"}
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item.get("text", "") for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""


def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


st.title("AI-Powered Tutor")
st.write("Ask questions from YouTube lecture transcripts.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("Transcript processed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.write("**Answer:**", answer)
import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled, 
    NoTranscriptFound,
    VideoUnavailable,                                                                                                        CouldNotRetrieveTranscript
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item["text"] for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""

def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

st.title("AI-Powered Tutor")
st.write("Ask questions from YouTube lecture transcripts.")

video_url = st.text_input("Enter YouTube Video URL")

import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        # Prefer get_transcript which returns a list of {"text","start","duration"}
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item.get("text", "") for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""


def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


st.title("AI-Powered Tutor")
st.write("Ask questions from YouTube lecture transcripts.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("Transcript processed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.write("**Answer:**", answer)
