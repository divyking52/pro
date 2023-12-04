from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
from pytube import YouTube
import os
import whisper
from transformers import GPT2LMHeadModel, GPT2Tokenizer


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
def save_video(url, video_filename):
    youtubeObject = YouTube(url)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("error occured while downloading")
    print("download completed")

    return video_filename

@app.route('/process_video', methods=['POST'])
def process_video():
    url = request.form['url']

    # Load GPT-2 model and tokenizer here
    gpt2_model, gpt2_tokenizer = load_gpt2_model()

    try:
        video_title, audio_filename, video_filename = save_audio(url)
        transcript_result = audio_to_transcription(audio_filename)
        recipe_result = generate_recipe_with_gpt2(transcript_result, gpt2_model, gpt2_tokenizer)

        video_url = f'/static/videos/{Path(video_filename).name}'

        return render_template('recipe_template.html', video_url=video_url, transcript_result=transcript_result, recipe_result=recipe_result)

    except Exception as e:
        # Provide user-friendly feedback in case of an error
        error_message = f"An error occurred: {str(e)}"
        return render_template('error_template.html', error_message=error_message)


def load_model():
    model = whisper.load_model("base")
    return model

def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file,file_name)

    audio_filename= Path(file_name).stem+'.mp3'
    video_filename = save_video(url,Path(file_name).stem+'.mp4')
    print(yt.title+ 'has been successfully downloaded')
    return yt.title, audio_filename, video_filename

def audio_to_transcription(audio_file):
    model = load_model()
    result= model.transcribe(audio_file)
    transcript = result['text']
    return transcript

def load_gpt2_model():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_recipe_with_gpt2(prompt, model, tokenizer):
    prompt= "write the structured food recipe from the below text:\n"+ prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=500,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            pad_token_id=tokenizer.eos_token_id)

    recipe = tokenizer.decode(output[0], skip_special_tokens=True)
    return recipe

if __name__ == '__main__':
    app.run(debug=True)


