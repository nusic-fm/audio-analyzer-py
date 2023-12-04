import io
import os
import librosa
import numpy as np
# import essentia.standard as es
from flask import Flask, send_file, request, jsonify
import matchering as mg
import tempfile
import allin1

app = Flask(__name__)
# Configure a temporary directory to store uploaded files
# UPLOAD_FOLDER = 'temp_uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    return jsonify({"text": "Hello world"}), 200

@app.route("/energy", methods=["POST"])
def librosa_energy_change():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # # Save the uploaded file to the temporary directory
        # temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        # print(temp_path)
        # file.save(temp_path)
        # return jsonify({"msg": "working"})
        try:
            # Load audio file
            # audio_file = "/content/audio.mp3"
            # y, sr = librosa.load(temp_path)
            audio_data = io.BytesIO(file.read())
            y, sr = librosa.load(audio_data)

            # Perform beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Compute onset envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Detect onsets
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)

            # Calculate time intervals between onsets
            intervals = onset_times[1:] - onset_times[:-1]

            # Analyze the distribution of intervals
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            # Define a function to dynamically adjust the threshold
            def dynamic_threshold(interval):
                return mean_interval + std_interval  # You can adjust this formula

            # Group onsets based on dynamic threshold
            grouped_onsets = []
            current_group = [onset_times[0]]

            for i, interval in enumerate(intervals):
                threshold = dynamic_threshold(interval)
                # print(threshold)
                if interval > threshold:
                    grouped_onsets.append(current_group)
                    current_group = [onset_times[i + 1]]
                else:
                    current_group.append(onset_times[i + 1])

            grouped_onsets.append(current_group)

            # Find the key using Essentia

            # Load the audio file
            # loader = es.MonoLoader(filename=audio_data)
            # audio = loader()

            # # Calculate the key
            # key_extractor = es.KeyExtractor()
            # key, scale, strength = key_extractor(audio)
            # , key=f"{key} {scale}"
            return jsonify(threshold=threshold, results = grouped_onsets, bpm=round(tempo), downbeat=beat_times[0])
        except Exception as e:
            return jsonify({'error': f'Error loaing file: {str(e)}'})

def process_audio(temp_path, ref_temp_path):
    # output_files = []
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # reference_file_path = os.path.join(current_directory, "reference.mp3")
    print("Initiating the Process")
        # Process the audio and save the results in the temporary directory
    mg.process(
        target=temp_path,
        reference=ref_temp_path,
        results=[
            # mg.pcm16("my_song_master_16bit.wav"),
            mg.pcm24("my_song_master_24bit.wav"),
        ],
    )
    
    # Collect the output files
    # output_files.append(os.path.join(current_directory, "my_song_master_16bit.wav"))
    # output_files.append(os.path.join(current_directory, "my_song_master_24bit.wav"))
    # return output_files
    return os.path.join(current_directory, "my_song_master_24bit.wav")

@app.route("/matchering", methods=["POST"])
def matchering():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    if 'referenceFile' not in request.files:
        return jsonify({'error': 'Reference File not provided, use referenceFile as param'})
    file = request.files['file']
    referenceFile = request.files['referenceFile']
    print(file.filename)

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded audio file to a temporary location
        temp_path = tempfile.mktemp(suffix='.wav')
        file.save(temp_path)
        # Save the uploaded audio file to a temporary location
        ref_temp_path = tempfile.mktemp(suffix='.mp3')
        referenceFile.save(ref_temp_path)
        print("Track is received and saved on the local", temp_path)
        
        try:
            output_file = process_audio(temp_path, ref_temp_path)
            return send_file(output_file, as_attachment=True)
        except Exception as e:
            return jsonify({'error': f'{str(e)}'})

@app.route("/all-in-one", methods=["POST"])
def allInOne():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    if 'referenceFile' not in request.files:
        return jsonify({'error': 'Reference File not provided, use referenceFile as param'})
    file = request.files['file']
    print(file.filename)
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Save the uploaded audio file to a temporary location
        temp_path = tempfile.mktemp(suffix='.wav')
        file.save(temp_path)
        print("Track is received and saved on the local", temp_path)
        
        try:
            result = allin1.analyze(temp_path, out_dir='./struct')
            
            return jsonify({'segments': result.segments, 'bpm': result.bpm, 'beats': result.beats, 'downbeats': result.downbeats, 'beat_positions': result.beat_positions})
        except Exception as e:
            return jsonify({'error': f'{str(e)}'})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))