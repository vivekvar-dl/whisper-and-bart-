# Audio & Video Transcription with Summarization

This project is a Streamlit application that allows users to transcribe and summarize audio and video files. It leverages state-of-the-art models from Hugging Face, including Whisper for transcription and RAG for summarization.

## Features

- **Audio Transcription:** Upload audio files in various formats (e.g., WAV, MP3, M4A) and transcribe them to text.
- **Video Transcription:** Upload video files (e.g., MP4, MOV), extract the audio, and transcribe it to text.
- **Summarization:** Summarize the transcribed text using RAG (Retrieval-Augmented Generation).
- **Multilingual Support:** Supports multiple languages for transcription, including English, Russian, Spanish, French, and German.

## Models Used

### Whisper Model
- **WhisperProcessor:** Pre-processes audio for transcription.
- **WhisperForConditionalGeneration:** Generates transcriptions from audio features.
- **Model Source:** [openai/whisper-base](https://huggingface.co/openai/whisper-base)

### RAG Model
- **RagTokenizer:** Tokenizes text for the RAG model.
- **RagRetriever:** Retrieves relevant documents for contextual generation.
- **RagSequenceForGeneration:** Generates text based on retrieved documents.
- **Model Source:** [facebook/rag-sequence-nq](https://huggingface.co/facebook/rag-sequence-nq)

## Installation

To run this application, you need to have Python and the required libraries installed. You can install the dependencies using `pip`.

```bash
pip install streamlit transformers torch soundfile librosa moviepy altair
