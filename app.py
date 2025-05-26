from flask import Flask, render_template, request, redirect, url_for,jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pyannote.audio import Pipeline,Model,Inference
import pandas as pd
import speech_recognition as sr
import regex as re 
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from pydub import AudioSegment
import glob
import shutil
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import torch
import os
import wave
import contextlib
import shutil
from pydub import AudioSegment
from collections import defaultdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor


app = Flask(__name__)
CORS(app) # to connect frontend with flask
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # for file uploads upto 1000mb

#GLobal varibales
UPLOAD_FOLDER = 'uploads'
VOICE_FOLDER='voices'
UPLOAD_USER='users'
SEGMENT_DIR = "segments"
segment_counter = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
result_df = pd.DataFrame(columns=["fileId", "speaker", "utterance","translation"])
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=os.getenv("hf_token")
)
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.getenv("hf_token"))
inference = Inference(embedding_model, window="whole")

# Speaker label storage
speaker_embeddings = []
segment_speakers = []
speaker_names = []

# openai key loading 
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

selected_language = "en-IN"
summary=False
executor = ThreadPoolExecutor(max_workers=5) # multiprocessing 
THRESHOLD = 0.8 
transcript_lines = []
unknown_speaker_count = 1
last_speaker = None

# Load IndicTrans model and tokenizer
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
config = BitsAndBytesConfig(load_in_8bit=True)
indic_en_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
ip = IndicProcessor(inference=True)


#functions

#diarization starts here     
def diarize_and_segment(chunk_path, rttm_path):
    global segment_counter, speaker_embeddings, segment_speakers, last_speaker, transcript_lines,unknown_speaker_count

    print(f" Diarizing {chunk_path}...")
    diarization = pipeline(chunk_path) # audio is sent to speaker-diarization-3.0
    print("Diarization result:", diarization)
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f) #saved as current audio file to rttm file 

    print(f" RTTM saved to {rttm_path}")
    audio = AudioSegment.from_wav(chunk_path)

    #rttm file is converted to dataframe
    df = pd.read_csv(rttm_path, sep=" ", header=None, comment=";", names=[
        "Type", "File ID", "Channel", "Start", "Duration",
        "NA1", "NA2", "Speaker", "NA3", "NA4"
    ])

    for _, row in df.iterrows():
        start = row["Start"]
        duration = row["Duration"]

        if duration < 0.5:
            continue

        buffer=800
        end = start + duration
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)+buffer 

        #using those rttm file data segments are created 
        segment_audio = audio[start_ms:end_ms]
        segment_filename = f"segment_{segment_counter}.wav"
        segment_path = os.path.join(SEGMENT_DIR, segment_filename) #saved in segments folder 
        segment_audio.export(segment_path, format="wav")
        print(f" Saved: {segment_filename} | Start={start:.3f}s Duration={duration:.3f}s")
        segment_counter += 1

        # Transcribe the segment
        transcript = extract_text_from_audio(segment_path, start_time=0, end_time=duration) #segments are sent to speech_recognition
        if not transcript or transcript.strip() == "":
            print(f" Skipping {segment_filename} — empty transcription")
            continue  # Skip embedding and labeling

        embeddings_root = "embeddings"
        wav_files = glob.glob(os.path.join(embeddings_root, "*.wav"))

        # Regex to extract speaker name from filename (e.g., nitik2.wav → nitik)
        speaker_pattern = re.compile(r"([a-zA-Z]+)")

        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            match = speaker_pattern.match(filename)
            if match:
                speaker_name = match.group(1)
                embedding = inference(wav_path).reshape(1, -1)
                speaker_embeddings.append(embedding)
                speaker_names.append(speaker_name)

        emb = inference(segment_path).reshape(1, -1)
        print(f'current={emb}')
        if not speaker_embeddings:
            speaker_embeddings.append(emb)
            speaker_label = "Speaker_1"
            print(f" {segment_filename} → {speaker_label} (first speaker)")
        else:
            distances = [cdist(emb, known_emb, metric="cosine")[0, 0] for known_emb in speaker_embeddings]
            min_dist = min(distances)

            if min_dist <= THRESHOLD:
                speaker_idx = distances.index(min_dist)
                speaker_label = speaker_names[speaker_idx]
            else:
                new_label = f"unknown_speaker_{unknown_speaker_count}"
                unknown_speaker_count += 1  # Increment for next unknown speaker
                speaker_embeddings.append(emb)
                speaker_names.append(new_label)
                speaker_label = new_label

            print(f"{segment_filename} → {speaker_label} (min_dist={min_dist:.4f})")
            segment_speakers.append((segment_filename, speaker_label))
            print(f"{speaker_label}: {transcript}")

        # Merge if same speaker as previous
        if speaker_label == last_speaker and transcript_lines:
            transcript_lines[-1] = transcript_lines[-1].strip() + f" {transcript}"
        else:
            transcript_lines.append(f"{speaker_label}: {transcript}")
        last_speaker = speaker_label

        # Write the updated transcript to file
        t_file="live.txt"
        with open(t_file, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#speech recognition 
def extract_text_from_audio(audio_file_path, start_time, end_time):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        start_ms = int(start_time * 1000)
        end_ms = int((end_time + 0.2) * 1000)
        segment = audio.get_segment(start_ms, end_ms)
        try:
            return recognizer.recognize_google(segment, language=selected_language)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            return f"API error: {e}" 
        
#indictrans2 get translation
def getTranslation(content):
    global selected_language
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if selected_language=="hi-IN":
            src_lang, tgt_lang = "hin_Deva", "eng_Latn"
        elif selected_language=="ta-IN":
            src_lang, tgt_lang = "tam_Taml", "eng_Latn"
        elif selected_language=="te-IN":
            src_lang, tgt_lang = "tel_Telu", "eng_Latn"
        elif selected_language=="bn-IN":
            src_lang, tgt_lang = "ben_Beng", "eng_Latn"
        elif selected_language=="gu-IN":
            src_lang, tgt_lang = "guj_Gujr", "eng_Latn"
        elif selected_language=="kn-IN":
            src_lang, tgt_lang = "kan_Knda", "eng_Latn"
        elif selected_language=="ml-IN":
            src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
        elif selected_language=="mr-IN":
            src_lang, tgt_lang = "mar_Deva", "eng_Latn"
        elif selected_language=="pa-IN":
            src_lang, tgt_lang = "pan_Guru", "eng_Latn"
        elif selected_language=="ur-IN":
            src_lang, tgt_lang = "urd_Arab", "eng_Latn"
        batch = content

        # Preprocess input
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize input
        inputs = indic_en_tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Generate translation
        with torch.no_grad():
            generated_tokens = indic_en_model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode output
        with indic_en_tokenizer.as_target_tokenizer():
            generated_tokens = indic_en_tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Post-process result
        en_translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        return en_translations
    except Exception as e:
        print(f"Error during translation: {e}")
        return f"Error: {e}"
        

@app.route('/')
def index():
    # Check if a file was uploaded (simple flag via query param)
    uploaded = request.args.get('uploaded')
    return render_template('index.html', uploaded=uploaded)

# audio files comes from frontend to this endpoint 
@app.route('/uploadchunk', methods=['POST'])
def upload_audio():
    print("Received upload request")
    if 'audio' not in request.files:
        print("No audio file in request")
        return jsonify({'error': 'No audio file'}), 400

    file = request.files['audio']
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(VOICE_FOLDER, filename) #saved in voices folder
    print(f"Saving to {filepath}")
    file.save(filepath)

    # Run diarization and segmentation
    rttm_path = filepath.replace(".wav", ".rttm")
    executor.submit(diarize_and_segment, filepath, rttm_path) #sent to this function with file and empty rttm file 

    return jsonify({'success': True, 'filename': filename}), 200

@app.route('/get_summary_live', methods=['GET'])
def get_summary_live():
    global summary_ready
    transcript_path = "live.txt"
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    else:
        full_text = ""

    if full_text.strip():
        prompt = f"""
        You are a helpful assistant. Please read the following meeting transcript and return the following:

        1. A complete summary of the conversation without missing any points
        2. Key discussion points (as bullet points) with which speaker said what
        3. Action items (as bullet points) which speaker should do what

        Transcript:
        {full_text}

        Format your response as:
        Summary: ...
        Key Points:
        - ...
        Actions:
        - ...
        """
        try:
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            content = response['choices'][0]['message']['content']
            summary_part = content.split("Key Points:")[0].replace("Summary:", "").strip()
            keypoints_part = content.split("Key Points:")[1].split("Actions:")[0].strip()
            actions_part = content.split("Actions:")[1].strip()

            app.config["SUMMARY"] = {
                "summary": summary_part,
                "key_points": keypoints_part,
                "actions": actions_part
            }
            summary_ready = True
            print("✅ Summary generation complete.")
        except Exception as e:
            print(f"❌ Error during summary generation: {e}")
            app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
            summary_ready = True
    else:
        print("⚠️ Empty transcript. No summary generated.")
        app.config["SUMMARY"] = {"summary": "", "key_points": "", "actions": ""}
        summary_ready = True
    
        

    return jsonify(app.config.get("SUMMARY", {
        "summary": "",
        "key_points": "",
        "actions": ""
    }))

@app.route('/get-translation')
def get_translation():
    transcript_path = "live.txt"
    if not os.path.exists(transcript_path):
        return jsonify({'translation': "Transcript not found."})

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
        content=[content]
    translation = getTranslation(content)
    return jsonify({'translation': translation})

@app.route('/get_transcript')
def get_transcript():
    transcript_path = "live.txt" 
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Filter out lines where the content is None or empty after colon
        cleaned_lines = [line.strip() for line in lines if ":" in line and line.strip().split(":", 1)[1].strip().lower() not in ["", "none"]]
        return jsonify({"transcript": "\n\n".join(cleaned_lines)})
    return jsonify({"transcript": ""})

@app.route('/clear_live', methods=['POST'])
def clearLive():
    global segment_counter,selected_language
    for folder in [VOICE_FOLDER, SEGMENT_DIR]:
        if os.path.exists(folder):
            print(f"Cleaning up {folder}: {os.listdir(folder)}")
            for f in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, f))
                except Exception as e:
                    print(f"Error deleting file {f}: {e}")
        else:
            print(f"Folder {folder} does not exist.")
    
    # Truncate live.txt instead of deleting it
    if os.path.exists("live.txt"):
        with open("live.txt", "w", encoding="utf-8") as f:
            f.truncate(0)
        print("live.txt truncated (emptied).")
    segment_counter=1
    selected_language='en-IN'
    return jsonify({'status': '✅ All chunks, segments, and transcript cleared'})

#file is uploaded from frontend to uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return jsonify({'success': True}), 200

# reads the file from uploads and process
@app.route('/process', methods=['POST'])
def process():
    wav_files = glob.glob("uploads/*.wav")

    if not wav_files:
        print("No .wav file found.")
        return jsonify({'success': False, 'message': 'No .wav file found.'}), 404

    filename = wav_files[0]
    print(f"Found file: {filename}")

    # Create FileChunks folder and clean it 
    os.makedirs("FileChunks", exist_ok=True)
    for f in os.listdir("FileChunks"):
        os.remove(os.path.join("FileChunks", f))

    # Load the full audio file and chunked for 6 second 
    audio = AudioSegment.from_file(filename)
    chunk_length_ms = 6000  # 6 seconds
    total_chunks = len(audio) // chunk_length_ms + int(len(audio) % chunk_length_ms > 0)

    # Save chunks
    for i in range(total_chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, len(audio))
        chunk = audio[start:end]
        chunk_filename = f"chunk_{i + 1}.wav"
        chunk_path = os.path.join("FileChunks", chunk_filename)
        chunk.export(chunk_path, format="wav")
        print(f"Saved: {chunk_path}")

        rttm_path = chunk_path.replace(".wav", ".rttm")
        #sent to diarize 
        diarize_and_segment(chunk_path, rttm_path)
        print(f"Diarization complete for: {chunk_path}")
    return jsonify({'success': True, 'chunks': total_chunks}), 200

@app.route('/clear', methods=['POST'])
def clear():
    global segment_counter,selected_language
    print("clear was clicked")

    # Clear .wav files from the uploads folder
    upload_folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if filename.endswith('.wav'):
            try:
                os.remove(file_path)
                print(f"Deleted {filename}")
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")

    # Clear contents of segments folder
    segments_folder = os.path.join(app.root_path, 'segments')
    if os.path.exists(segments_folder):
        for filename in os.listdir(segments_folder):
            file_path = os.path.join(segments_folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted from segments: {filename}")
            except Exception as e:
                print(f"Error deleting from segments {filename}: {e}")

    # Clear contents of FileChunks folder
    chunks_folder = os.path.join(app.root_path, 'FileChunks')
    if os.path.exists(chunks_folder):
        for filename in os.listdir(chunks_folder):
            file_path = os.path.join(chunks_folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted from FileChunks: {filename}")
            except Exception as e:
                print(f"Error deleting from FileChunks {filename}: {e}")

    # Truncate transcript.txt
    if os.path.exists("live.txt"):
        with open("live.txt", "w", encoding="utf-8") as f:
            f.truncate(0)
        print("live.txt truncated (emptied).")
    segment_counter=1
    selected_language='en-IN'
    return "Files cleared", 200

@app.route('/set_language', methods=['POST'])
def set_language():
    global selected_language
    data = request.get_json()
    selected_language = data.get('language', 'en-IN')
    print("🔤 Language set to:", selected_language)
    return jsonify({'success': True})

