# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from flask import Flask, render_template, request, send_file, Response
import urllib.request
import urllib.parse
import io

app = Flask(__name__)

# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/search")
def search():
    return render_template("search.html")

# ── TEXT-TO-SPEECH PROXY ─────────────────────────────────────────────────────

@app.route("/tts")
def tts():
    """Server-side TTS proxy — fetches audio from Google Translate for any language."""
    text = request.args.get("text", "")
    lang = request.args.get("lang", "en")
    if not text:
        return "No text", 400
    try:
        encoded = urllib.parse.quote(text)
        url = (
            "https://translate.google.com/translate_tts"
            "?ie=UTF-8"
            "&tl=" + lang +
            "&client=tw-ob"
            "&q=" + encoded
        )
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        req.add_header("Referer", "https://translate.google.com/")
        resp = urllib.request.urlopen(req, timeout=10)
        audio_data = resp.read()
        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as e:
        print(f"TTS error: {e}")
        return str(e), 500

# ── DISEASE INFO MAP ─────────────────────────────────────────────────────────

DISEASE_INFO = {
    'Bacterial': {
        'display': 'Bacterial Spot',
        'cause': 'Caused by Xanthomonas bacteria. It creates small, water-soaked spots on leaves that turn dark brown or black.',
        'rem': 'Use copper-based fungicides. Avoid overhead watering and rotate crops every year.',
        'rem1': ['Remove infected plants', 'Use copper spray', 'Crop rotation'],
        'supplement': '/static/supplements/copper_fungicide.png',
        'buy_url': 'https://www.amazon.in/s?k=copper+fungicide'
    },
    'Mosaic_virus': {
        'display': 'Mosaic Virus',
        'cause': 'Causes mottled green and yellow patterns. Leaves may look twisted or small. Spread by aphids and tools.',
        'rem': 'Control aphids using Neem Oil. Remove infected plants immediately. Disinfect tools frequently.',
        'rem1': ['Apply Neem Oil', 'Remove infected leaves', 'Tool disinfection'],
        'supplement': '/static/supplements/neem_oil.png',
        'buy_url': 'https://www.amazon.in/s?k=neem+oil'
    },
    'Spectoria': {
        'display': 'Septoria Leaf Spot',
        'cause': 'Common fungal disease causing small circular spots with dark borders and grey centers.',
        'rem': 'Apply fungicides like Daconil. Mulch around plants to prevent spores from splashing up.',
        'rem1': ['Daconil fungicide', 'Proper mulching', 'Air circulation'],
        'supplement': '/static/supplements/daconil_fungicide.png',
        'buy_url': 'https://www.amazon.in/s?k=daconil'
    },
    'Leafmod': {
        'display': 'Leaf Mold',
        'cause': 'A fungal disease that thrives in high humidity. You will see yellow spots on top and olive-green mold underneath.',
        'rem': 'Improve air circulation. Use fungicides containing Chlorothalonil or Copper. Avoid wetting leaves.',
        'rem1': ['Reduce humidity', 'Apply Chlorothalonil', 'Pruning'],
        'supplement': '/static/supplements/neem_oil.png',
        'buy_url': 'https://www.amazon.in/s?k=chlorothalonil'
    },
    'curl_virus': {
        'display': 'Yellow Leaf Curl Virus',
        'cause': 'Transmitted by whiteflies. Leaves curl upward and turn yellow. Plants become stunted and produce no fruit.',
        'rem': 'Use yellow sticky traps for whiteflies. Apply Neem oil. Use virus-resistant varieties.',
        'rem1': ['Yellow sticky traps', 'Neem oil spray', 'Resistant varieties'],
        'supplement': '/static/supplements/sticky_traps.png',
        'buy_url': 'https://www.amazon.in/s?k=yellow+sticky+traps'
    },
    'Healthy': {
        'display': 'Healthy Leaf',
        'cause': 'Your plant looks great! No disease detected.',
        'rem': 'Maintain good watering habits and monitor for pests regularly.',
        'rem1': ['Regular watering', 'Sunlight exposure', 'Nutrient soil'],
        'supplement': '/static/supplements/plant_fertilizer.png',
        'buy_url': 'https://www.amazon.in/s?k=plant+fertilizer'
    }
}

LABEL_MAP = {0: "Healthy", 1: "Bacterial", 2: "Mosaic_virus",
             3: "Spectoria", 4: "Leafmod",  5: "curl_virus"}

# ── IMAGE / PREDICTION ROUTE ─────────────────────────────────────────────────

@app.route("/image", methods=["GET", "POST"])
def image():
    if request.method != "POST":
        return render_template("index.html")

    # 1. Clear previous images
    dir_path = "static/images"
    os.makedirs(dir_path, exist_ok=True)
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))

    # 2. Save uploaded file
    file      = request.files["filename"]
    file_name = file.filename
    save_path = os.path.join(dir_path, file_name)
    file.save(save_path)

    # 3. Constants
    IMG_SIZE   = 50
    LR         = 1e-3
    MODEL_NAME = "healthyvsunhealthynew-{}-{}.model".format(LR, "2conv-basic")

    # 4. Load and preprocess image
    def load_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image: " + path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return np.array(img)

    try:
        img_array = load_image(save_path)
    except ValueError as e:
        return render_template("index.html", error=str(e))

    # 5. Build model
    tf.compat.v1.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name="input")
    convnet = conv_2d(convnet, 32,  3, activation="relu")
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 64,  3, activation="relu")
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 128, 3, activation="relu")
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 32,  3, activation="relu")
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 64,  3, activation="relu")
    convnet = max_pool_2d(convnet, 3)
    convnet = fully_connected(convnet, 1024, activation="relu")
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 6, activation="softmax")
    convnet = regression(convnet, optimizer="adam", learning_rate=LR,
                         loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(convnet, tensorboard_dir="log")

    if os.path.exists("{}.meta".format(MODEL_NAME)):
        model.load(MODEL_NAME)
    
    # 6. Predict
    data_arr  = img_array.reshape(IMG_SIZE, IMG_SIZE, 3)
    model_out = model.predict([data_arr])[0]
    pred_idx  = int(np.argmax(model_out))
    str_label = LABEL_MAP.get(pred_idx, "Unknown")

    # 7. Fetch disease info
    info = DISEASE_INFO.get(str_label, {
        "display":    "Unknown",
        "cause":      "Could not determine disease. Please try a clearer image.",
        "rem":        "",
        "rem1":       [],
        "supplement": "",
        "buy_url":    "https://www.amazon.in/s?k=plant+disease+treatment",
    })

    # 8. Render
    return render_template(
        "index.html",
        status       = str_label,
        disease      = info["display"],
        causes       = info["cause"],
        remedie      = info["rem"],
        remedie1     = info["rem1"],
        supplements  = info["supplement"],
        buy_link     = info.get("buy_url", "https://www.amazon.in/s?k=plant+disease+treatment"),
        ImageDisplay = "http://127.0.0.1:5000/static/images/" + file_name,
    )

if __name__ == "__main__":
    app.run(debug=True)