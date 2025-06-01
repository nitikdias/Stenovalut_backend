from flask import Flask, render_template, request, redirect, url_for
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
import sys
import requests
import json
import base64
import torchaudio
import time


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
app.config['VOICE_FOLDER'] = VOICE_FOLDER
result_df = pd.DataFrame(columns=["fileId", "speaker", "utterance","translation"])
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="change_here"
)
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token="change_here")
inference = Inference(embedding_model, window="whole")

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",use_auth_token="change_here")

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
translation_lines=[]
unknown_speaker_count = 1
last_speaker = None

# indictrans2
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

#bhashini
API_KEY = "Bhashini_api_change_here"
USER_ID = "change_here_id"
CONFIG_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"


# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VOICE_FOLDER, exist_ok=True)
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
        
#bhashini

def get_pipeline_config():
    """Get pipeline configuration and callback URL"""
    headers = {
        "ulcaApiKey": API_KEY,
        "userID": USER_ID,
        "Content-Type": "application/json"
    }
    
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": "en"
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": "64392f96daac500b55c543cd"
        }
    }
    
    response = requests.post(CONFIG_URL, headers=headers, json=payload)
    print("Config API Status:", response.status_code)
    print("Config API Response:", response.text)
    
    if response.ok:
        config_data = response.json()
        callback_url = config_data.get("pipelineInferenceAPIEndPoint", {}).get("callbackUrl")
        auth_key = config_data.get("pipelineInferenceAPIEndPoint", {}).get("inferenceApiKey", {}).get("name")
        auth_value = config_data.get("pipelineInferenceAPIEndPoint", {}).get("inferenceApiKey", {}).get("value")
        
        # Get the service ID from the response
        asr_service_id = None
        pipeline_response_config = config_data.get("pipelineResponseConfig", [])
        for task in pipeline_response_config:
            if task.get("taskType") == "asr":
                configs = task.get("config", [])
                if configs:
                    asr_service_id = configs[0].get("serviceId")
                break
        
        return callback_url, auth_key, auth_value, asr_service_id
    else:
        print("Failed to get pipeline config")
        return None, None, None, None
    

def transcribe_audio(audio_path):
    # First, get the pipeline configuration
    callback_url, auth_key, auth_value, service_id = get_pipeline_config()
    
    if not callback_url:
        print("Failed to get callback URL from config")
        return
    
    print(f"Using callback URL: {callback_url}")
    print(f"Using service ID: {service_id}")
    
    # Prepare headers for inference request
    headers = {
        "Content-Type": "application/json"
    }
    if auth_key and auth_value:
        headers[auth_key] = auth_value
    
    # Read and encode audio file
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    print(f"Audio file size: {len(audio_base64)} characters in base64")
    print(f"Approximate file size: {len(audio_base64) * 3 / 4 / 1024 / 1024:.2f} MB")
    
    # Check if file is too large (limit to ~5MB)
    if len(audio_base64) * 3 / 4 > 5 * 1024 * 1024:
        print("Warning: Audio file might be too large. Consider using a shorter audio clip.")
    
    # Prepare inference payload
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": "en"},
                    "serviceId": service_id
                }
            }
        ],
        "inputData": {
            "audio": [
                {"audioContent": audio_base64}
            ]
        }
    }
    
    # Make inference request
    response = requests.post(callback_url, headers=headers, json=payload)
    print("Inference Status Code:", response.status_code)
    print("Inference Response:", response.text)
    
    if response.ok:
        result = response.json()
        # Extract transcription from response
        try:
            transcription = result.get("pipelineResponse", [{}])[0].get("output", [{}])[0].get("source", "No transcription found")
            print(f"\nTranscription: {transcription}")
            return transcription  # ← ADD THIS LINE
        except (IndexError, KeyError):
            print("Could not extract transcription from response")
            print("Full response:", json.dumps(result, indent=2))
            return None  # ← OPTIONAL fallback return
    else:
        print("Error:", response.status_code, response.text)
        return None  # ← OPTIONAL fallback return

         
#diarization starts here     
def safe_transcribe_audio(path, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            result = transcribe_audio(path)
            if result and result.strip().lower() != "none":
                return result
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Error during transcription: {e}")
        time.sleep(delay)
    print(f"Failed to transcribe {path} after {max_retries} attempts.")
    return ""

def diarize_and_segment(chunk_path, rttm_path):
    global segment_counter, speaker_embeddings, segment_speakers, last_speaker, transcript_lines, unknown_speaker_count, selected_language

    print(f" Diarizing {chunk_path}...")
    diarization = pipeline(chunk_path)
    print("Diarization result:", diarization)

    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)

    print(f" RTTM saved to {rttm_path}")
    audio = AudioSegment.from_wav(chunk_path)

    df = pd.read_csv(rttm_path, sep=" ", header=None, comment=";", names=[
        "Type", "File ID", "Channel", "Start", "Duration",
        "NA1", "NA2", "Speaker", "NA3", "NA4"
    ])

    for _, row in df.iterrows():
        start = row["Start"]
        duration = row["Duration"]

        if duration < 0.5:
            continue

        buffer = 800
        end = start + duration
        start_ms = int(start * 1000)
        end_ms = int(end * 1000) + buffer

        segment_audio = audio[start_ms:end_ms]
        segment_filename = f"segment_{segment_counter}.wav"
        segment_path = os.path.join(SEGMENT_DIR, segment_filename)
        segment_audio.export(segment_path, format="wav")
        print(f" Saved: {segment_filename} | Start={start:.3f}s Duration={duration:.3f}s")
        segment_counter += 1

        # Transcribe and translate sequentially
        if selected_language != "en-IN":
            transcript = extract_text_from_audio(segment_path, 0, duration)
            translate = safe_transcribe_audio(segment_path)
        else:
            transcript = extract_text_from_audio(segment_path, 0, duration)
            translate = transcript

        # Clean up None or empty strings
        transcript = transcript or ""
        translate = translate if translate and translate.strip().lower() != "none" else ""

        if transcript.strip() == "":
            print(f"Skipping {segment_filename} — empty transcription")
            continue

        # Speaker embeddings
        embeddings_root = "embeddings"
        wav_files = glob.glob(os.path.join(embeddings_root, "*.wav"))
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
            print(f"{segment_filename} → {speaker_label} (first speaker)")
        else:
            distances = [cdist(emb, known_emb, metric="cosine")[0, 0] for known_emb in speaker_embeddings]
            min_dist = min(distances)

            if min_dist <= THRESHOLD:
                speaker_idx = distances.index(min_dist)
                speaker_label = speaker_names[speaker_idx]
            else:
                new_label = f"unknown_speaker"
                unknown_speaker_count += 1
                speaker_embeddings.append(emb)
                speaker_names.append(new_label)
                speaker_label = new_label

            print(f"{segment_filename} → {speaker_label} (min_dist={min_dist:.4f})")

        # Append transcript and translation
        if speaker_label == last_speaker and transcript_lines:
            transcript_lines[-1] += f" {transcript}"
            translation_lines[-1] += f" {translate}"
        else:
            transcript_lines.append(f"{speaker_label}: {transcript}")
            translation_lines.append(f"{speaker_label}: {translate}")

        last_speaker = speaker_label

        # Write to files
        with open("live_transcript.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(transcript_lines))

        with open("live_translation.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(translation_lines))

#for live
def diarize_and_segment_live(chunk_path, rttm_path):
    global segment_counter, speaker_embeddings, segment_speakers, last_speaker, transcript_lines,unknown_speaker_count,selected_language

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
        if selected_language=="ta-IN":
            transcript = transcribe_audio(segment_path)
        else:
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
                print(THRESHOLD)
                speaker_idx = distances.index(min_dist)
                speaker_label = speaker_names[speaker_idx]
            else:
                new_label = f"unknown_speaker"
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
        t_file="live_transcript.txt"
        with open(t_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(transcript_lines))



#indictrans2 get translation
def getTranslation(content):
    def initialize_model_and_tokenizer(ckpt_dir, quantization):
        if quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        )

        if qconfig == None:
            model = model.to(DEVICE)
            if DEVICE == "cuda":
                model.half()

        model.eval()

        return tokenizer, model


    def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
        translations = []
        for i in range(0, len(input_sentences), BATCH_SIZE):
            batch = input_sentences[i : i + BATCH_SIZE]

            # Preprocess the batch and extract entity mappings
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

            # Tokenize the batch and generate input encodings
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode the generated tokens into text

            with tokenizer.as_target_tokenizer():
                generated_tokens = tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            # Postprocess the translations, including entity replacement
            translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations

    indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"  # ai4bharat/indictrans2-indic-en-dist-200M
    indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir, quantization)

    ip = IndicProcessor(inference=True)
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
    en_translations = batch_translate(content, src_lang, tgt_lang, indic_en_model, indic_en_tokenizer, ip)
    

    print(f"\n{src_lang} - {tgt_lang}")
    for input_sentence, translation in zip(content, en_translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")
    with open("live_translation.txt", "w", encoding="utf-8") as f:
        f.write('\n'.join(en_translations))

 
    
    return en_translations

def vad(file):
    buffer_seconds = 0.3  # buffer length in seconds

    # Run VAD pipeline on audio file, returns timeline of speech segments
    vad_result = pipeline(file)

    # Load full audio waveform and sample rate
    waveform, sample_rate = torchaudio.load(file)
    audio_duration = waveform.shape[1] / sample_rate

    output_dir = "FileChunks"  # keep consistent with route
    os.makedirs(output_dir, exist_ok=True)

    chunk_paths = []

    for i, segment in enumerate(vad_result.get_timeline()):
        # Apply buffer, clipped to audio duration
        start = max(segment.start - buffer_seconds, 0)
        end = min(segment.end + buffer_seconds, audio_duration)

        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        chunk_waveform = waveform[:, start_sample:end_sample]

        output_path = os.path.join(output_dir, f"chunk_{i + 1}.wav")
        torchaudio.save(output_path, chunk_waveform, sample_rate)
        print(f"Saved: {output_path} (from {start:.2f}s to {end:.2f}s)")

        chunk_paths.append(output_path)

    return chunk_paths

    

@app.route('/')
def index():
    # Check if a file was uploaded (simple flag via query param)
    uploaded = request.args.get('uploaded')
    return render_template('index.html', uploaded=uploaded)

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


from flask import jsonify
# reads the file from uploads and process
@app.route('/process', methods=['POST'])
def process():
    global THRESHOLD
    THRESHOLD=0.687
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

    # Call your VAD function here which returns list of chunk paths
    chunk_paths = vad(filename)  # This should save chunks inside "FileChunks" and return their paths
    
    if not chunk_paths:
        print("No speech segments found by VAD.")
        return jsonify({'success': False, 'message': 'No speech detected.'}), 400

    # Process each chunk with diarization
    for chunk_path in chunk_paths:
        rttm_path = chunk_path.replace(".wav", ".rttm")
        diarize_and_segment(chunk_path, rttm_path)
        print(f"Diarization complete for: {chunk_path}")

    return jsonify({'success': True, 'chunks': len(chunk_paths)}), 200





@app.route('/clear', methods=['POST'])
def clear():
    global segment_counter, selected_language, speaker_embeddings, segment_speakers, speaker_names, transcript_lines, translation_lines, unknown_speaker_count,last_speaker, summary
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
    if os.path.exists("live_transcript.txt"):
        with open("live_transcript.txt", "w", encoding="utf-8") as f:
            f.truncate(0)
        print("live_transcript.txt truncated (emptied).")
    if os.path.exists("live_translation.txt"):
        with open("live_translation.txt", "w", encoding="utf-8") as f:
            f.truncate(0)
        print("live_translation.txt truncated (emptied).")
    segment_counter = 1
    selected_language = 'en-IN'
    speaker_embeddings = []
    segment_speakers = []
    speaker_names = []
    transcript_lines = []
    translation_lines = []
    unknown_speaker_count = 1
    last_speaker = None
    summary = False
    return "Files cleared", 200



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
    executor.submit(diarize_and_segment_live, filepath, rttm_path) #sent to this function with file and empty rttm file 

    return jsonify({'success': True, 'filename': filename}), 200


@app.route('/transcribeUser')
def transcribe():
    try:
        audio="users/newuser.wav"
        text = transcribe_audio(audio)
        print(text)

        # Extract name between "my name is" and "the curious engineer"
        name_match = re.search(r'my name is (.*?) the', text, re.IGNORECASE)
        name = name_match.group(1).strip().rstrip('.') if name_match else ""

        return jsonify({'success': True, 'text': text, 'name': name})

    except sr.UnknownValueError:
        return jsonify({'success': False, 'error': 'Could not understand audio'})
    except sr.RequestError as e:
        return jsonify({'success': False, 'error': str(e)})


    
@app.route('/upload-User', methods=['POST'])
def upload_user():
    if 'audio' not in request.files:
        return jsonify({'message': '❌ No file uploaded'}), 400

    audio = request.files['audio']
    save_path = os.path.join(UPLOAD_USER, 'newuser.wav')
    audio.save(save_path)
    return jsonify({'message': '✅ Audio saved as newuser.wav'})

@app.route('/set_language', methods=['POST'])
def set_language():
    global selected_language
    data = request.get_json()
    selected_language = data.get('language', 'en-IN')
    print("🔤 Language set to:", selected_language)
    return jsonify({'success': True})

@app.route('/get_transcript', methods=['GET'])
def get_transcript():
    global speaker_embeddings
    with open("live_transcript.txt", "r", encoding="utf-8") as f:
        transcript = f.read()
    try:
        with open("live_translation.txt", "r", encoding="utf-8") as f:
            translation = f.read()
    except FileNotFoundError:
        translation = ""

    return jsonify({
        "transcript": transcript,
        "translation": translation
    })
    


@app.route('/get_summary_live', methods=['GET'])
def get_summary_live():
    global summary_ready,selected_language
    if selected_language=="en-IN":
        transcript_path = "live_transcript.txt"
    else:
        transcript_path="live_translation.txt"
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

@app.route('/clear_live', methods=['POST'])
def clearLive():
    global segment_counter, selected_language, speaker_embeddings, segment_speakers, speaker_names, transcript_lines, translation_lines, unknown_speaker_count, last_speaker, summary

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
    
    # Truncate transcript.txt and translation.txt
    for file_path in ["live_transcript.txt", "live_translation.txt"]:
        if os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.truncate(0)
            print(f"{file_path} truncated (emptied).")

    # Reset all global variables to initial state
    segment_counter = 1
    selected_language = 'en-IN'
    speaker_embeddings = []
    segment_speakers = []
    speaker_names = []
    transcript_lines = []
    translation_lines = []
    unknown_speaker_count = 1
    last_speaker = None
    summary = False

    print("All global variables reset.")

    return jsonify({'status': '✅ All chunks, segments, and transcript cleared'})


@app.route('/register-speaker', methods=['POST'])
def register_speaker():
    data = request.get_json()
    name = data.get('name')

    if not name:
        return jsonify({'success': False, 'error': 'Missing name'})

    source_path = os.path.join("users", "newuser.wav")
    dest_path = os.path.join("embeddings", f"{name}.wav")
    shutil.copy(source_path, dest_path)

    # Compute and store embedding in memory
    embedding = inference(dest_path).reshape(1, -1)
    speaker_embeddings.append(embedding)
    speaker_names.append(name)

    return jsonify({'success': True, 'message': 'Speaker registered successfully'})

    
@app.route('/get-translation')
def get_translation():
    transcript_path = "live_transcript.txt"
    if not os.path.exists(transcript_path):
        return jsonify({'translation': "Transcript not found."})

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
        content=[content]
    translation = getTranslation(content)
    return jsonify({'translation': translation})

@app.route('/get-translation-file')
def get_translation_file():
    transcript_path = "transcript.txt"
    if not os.path.exists(transcript_path):
        return jsonify({'translation': "Transcript not found."})

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
        content=[content]
    translation = getTranslation(content)
    return jsonify({'translation': translation})

EMBEDDINGS_DIR = 'embeddings'  # path to your embeddings folder

@app.route('/list-users', methods=['GET'])
def list_users():
    try:
        # List all .wav files in the embeddings folder
        files = os.listdir(EMBEDDINGS_DIR)
        audio_files = [f for f in files if f.lower().endswith('.wav')]

        # Strip .wav extension
        user_names = [os.path.splitext(f)[0] for f in audio_files]

        return jsonify({"users": user_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
