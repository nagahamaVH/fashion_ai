import base64
import os
import re
from flask import request, Flask, jsonify
from PIL import Image
from train import train as train
from predict import predict
from utils import decode_image


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "Fashion AI"


@app.route("/train", methods=["GET"])
def train_api():
    model_name = request.args.get("model_name", type=str)
    num_epochs = request.args.get("num_epochs", 10, type=int)
    batch_size = request.args.get("batch_size", 3, type=int)
    valid_size = request.args.get("test_size", 0.2, type=float)
    seed = request.args.get("seed", 259, type=int)
    device = request.args.get("device", "none", type=str)
    num_workers = request.args.get("num_workers", -1, type=int)
    use_wandb = request.args.get("use_wandb", 0, type=int)
    project_name = request.args.get("project_name", "fashion_ai", type=str)
    entity = request.args.get("entity", "nagahamavh", type=str)
    use_tqdm = request.args.get("use_tqdm", 0, type=int)

    if device == "none":
        device = None
    use_wandb = True if use_wandb == 1 else False
    use_tqdm = True if use_tqdm == 1 else False

    train(model_name, num_epochs=num_epochs, batch_size=batch_size,
          valid_size=valid_size, seed=seed, device=device,
          num_workers=num_workers, use_wandb=use_wandb,
          project_name=project_name, entity=entity, use_tqdm=use_tqdm)

    return "Modelo " + model_name + " treinado com sucesso"


@app.route("/predict", methods=["GET"])
def predict_api():
    received_image = request.args.get('image')
    model_name = request.args.get('model_name', type=str)
    device = request.args.get("device", "none", type=str)

    if device == "none":
        device = None

    image = decode_image(received_image)
    image, info = predict(image, model_name, device)

    image_pil = Image.fromarray(image)
    image_pil.save("output/new_image.png")

    with open("output/new_image.png", "rb") as f:
        sended_image = base64.b64encode(f.read()).decode("utf-8")

    return jsonify({"image": sended_image, "info": info})


@app.route("/get_models", methods=["GET"])
def get_models():
    all_files = os.listdir("./pretrained_models")
    all_files = [re.sub(r'\.\w+$', '', string) for string in all_files]
    models = list(set(all_files))

    return jsonify(models)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
