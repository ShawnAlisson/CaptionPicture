from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time
from model.mobilenet_model import generate_caption_mobilenet
from model.inception_model import generate_caption_inception
from model.resnet_model import generate_caption_resnet
from model.transformer_model import generate_caption_transformer
from model.blip_model import generate_caption_blip

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'file' not in request.files or 'model' not in request.form:
        return redirect(url_for('index'))
    
    file = request.files['file']
    selected_model = request.form['model']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        start_time = time.time()
        if selected_model == 'mobilenet':
            caption = generate_caption_mobilenet(filepath)
        elif selected_model == 'inception':
            caption = generate_caption_inception(filepath)
        elif selected_model == 'resnet':
            caption = generate_caption_resnet(filepath)
        elif selected_model == 'transformer':
            caption = generate_caption_transformer(filepath)
        elif selected_model == 'blip':
            caption = generate_caption_blip(filepath)
        end_time = time.time()
        
        processing_time = end_time - start_time
        os.remove(filepath)  # Delete the file after captioning
        return render_template('index.html', image_url=filepath, caption=caption, model=selected_model, time=processing_time)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=8000)
