import os
import pandas as pd

from helpers.chunking import split_audio_to_chunks
from helpers.gcs_upload import upload_to_gcs
from helpers.transcription import transcribe_audio_with_word_timestamps
from helpers.sentence_align import split_into_sentences, assign_timestamps_to_sentences
from helpers.sentiment import analyze_sentiment
from helpers.load_phrases import load_phrases_from_file

BUCKET_NAME = "simpliearn-audio"
AUDIO_FOLDER = "input_audio"

def run_pipeline(input_file):
    filename = os.path.splitext(os.path.basename(input_file))[0]
    company_name = filename.split("_")[0].lower()

    print(f"\n--- Processing {filename} ---")
    print("Loading phrases...")
    phrase_file = os.path.join("phrases", f"{company_name}.txt")
    phrases = load_phrases_from_file(phrase_file)

    print("Splitting audio...")
    chunk_subfolder = os.path.join("chunks", company_name)
    chunks_metadata = split_audio_to_chunks(input_file, "chunks", chunk_length_sec=300, subfolder=company_name)

    print("Uploading to GCS...")
    for chunk in chunks_metadata:
        gcs_uri = upload_to_gcs(
            BUCKET_NAME,
            chunk["path"],  # full path from split_audio_to_chunks
            chunk["filename"],
            folder_prefix=filename  # folder in GCS: e.g., 'tesla_q4/tesla_q4_chunk_000.mp3'
        )
        chunk["gcs_uri"] = gcs_uri

    print("Transcribing audio...")
    full_transcript = ""
    all_word_timestamps = []

    for chunk in chunks_metadata:
        print(f"Transcribing {chunk['filename']}...")
        transcript, word_times = transcribe_audio_with_word_timestamps(chunk["gcs_uri"], phrases)

        for word in word_times:
            word["start_time"] += chunk["start_time"]
            word["end_time"] += chunk["start_time"]

        full_transcript += transcript.strip() + " "
        all_word_timestamps.extend(word_times)

    print("Splitting into sentences...")
    sentences = split_into_sentences(full_transcript)
    matched_sentences = assign_timestamps_to_sentences(sentences, all_word_timestamps)

    print("Analyzing sentiment...")
    sentiment_results = analyze_sentiment(matched_sentences)

    print("Saving results...")
    output_dir = os.path.join("saved_data", company_name)
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"{filename}_sentiment.csv")
    pd.DataFrame(sentiment_results).to_csv(output_csv, index=False)

    print(f"Done. Sentiment data saved to {output_csv}")

if __name__ == "__main__":
    for file in os.listdir(AUDIO_FOLDER):
        if file.endswith(".mp3"):
            input_path = os.path.join(AUDIO_FOLDER, file)
            run_pipeline(input_path)
