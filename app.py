import streamlit as st
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    RagTokenizer, 
    RagRetriever, 
    RagSequenceForGeneration
)
import torch
import soundfile as sf
import librosa
from moviepy.editor import VideoFileClip
import tempfile
import altair as alt

# Load pre-trained models and processors
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
rag_retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=rag_retriever)

# Function to transcribe audio
def transcribe_audio(audio_path, language="en"): 
    speech, _ = librosa.load(audio_path, sr=16000)
    input_features = whisper_processor(speech, return_tensors="pt", sampling_rate=16000).input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Function to translate and summarize text (simplified for demonstration)
def translate_and_summarize(text):
    input_ids = rag_tokenizer(text, return_tensors="pt").input_ids
    output_ids = rag_model.generate(input_ids)
    summary = rag_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return summary

# Function to extract audio from video
def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    if video.audio:
        video.audio.write_audiofile(output_audio_path)
        return output_audio_path

# Streamlit App
st.title("Audio & Video Transcription with Summarization")

# Audio Transcription Section
st.header("Transcribe Audio")
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
language = st.selectbox("Select Language", ["en", "ru", "es", "fr", "de"])  # Add more languages as needed

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_file.read())
        st.audio(audio_file, format="audio/wav")
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(tmp_audio.name, language=language)
        st.subheader("Transcription:")
        st.write(transcription)
        with st.spinner("Summarizing..."):
            summary = translate_and_summarize(transcription)  # Replace with more robust summarization logic
        st.subheader("Summary:")
        st.write(summary)


# Video Transcription Section
st.header("Transcribe Video")
video_file = st.file_uploader("Upload Video File", type=["mp4", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        st.video(video_file, format="video/mp4")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            audio_extracted = extract_audio_from_video(tmp_video.name, tmp_audio.name)
            if audio_extracted:
                with st.spinner("Transcribing..."):
                    transcription = transcribe_audio(tmp_audio.name, language=language) 
                st.subheader("Transcription:")
                st.write(transcription)
                with st.spinner("Summarizing..."):
                    summary = translate_and_summarize(transcription)
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.write("No audio track found in the video.")
