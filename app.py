from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename
import os
import base64
from io import BytesIO
import torch.nn.functional as F

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 2MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_class, model_path, num_classes):
    try:
        model = model_class(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

from model_cbam import MobileNetV2_CBAM, EfficientNetB0_CBAM, ShuffleNetV2_CBAM, MobileNetV2_NoCBAM, EfficientNetB0_NoCBAM, ShuffleNetV2_NoCBAM

# 8-class models
model_paths = {
    "MobileNetV2_NoCBAM": "D:/Deep Learning/Web_App Varietas Jagung/model/full_best_testNew_MobileNetV2.pth",
    "EfficientNetB0_NoCBAM": "D:/Deep Learning/Web_App Varietas Jagung/model/full_best_testNew_EfficientNetB0.pth",
    "ShuffleNetV2_NoCBAM": "D:/Deep Learning/Web_App Varietas Jagung/model/full_best_testNew_ShuffleNetV2.pth",
}

# Load semua model
models = {
    "MobileNetV2_NoCBAM": load_model(MobileNetV2_NoCBAM, model_paths["MobileNetV2_NoCBAM"], 8),     
    "EfficientNetB0_NoCBAM": load_model(EfficientNetB0_NoCBAM, model_paths["EfficientNetB0_NoCBAM"], 8),
    "ShuffleNetV2_NoCBAM": load_model(ShuffleNetV2_NoCBAM, model_paths["ShuffleNetV2_NoCBAM"], 8),
}

# Reclassification models (3 classes only)
reclass_paths = {
    "MobileNetV2_CBAM": "D:/Deep Learning/Web_App Varietas Jagung/model/final_testNew_MobileNetV2.pth",
    "EfficientNetB0_CBAM": "D:/Deep Learning/Web_App Varietas Jagung/model/final_testNew_EfficientNetB0.pth",
    "ShuffleNetV2_CBAM": "D:/Deep Learning/Web_App Varietas Jagung/model/final_testNew_ShuffleNetV2.pth"
}

reclass_models = {
    "MobileNetV2_CBAM": load_model(MobileNetV2_NoCBAM, reclass_paths["MobileNetV2_CBAM"], 2),
    "EfficientNetB0_CBAM": load_model(EfficientNetB0_NoCBAM, reclass_paths["EfficientNetB0_CBAM"], 2),
    "ShuffleNetV2_CBAM": load_model(ShuffleNetV2_NoCBAM, reclass_paths["ShuffleNetV2_CBAM"], 2),
}

if not all(models.values()) or not all(reclass_models.values()):
    raise RuntimeError("Gagal memuat satu atau lebih model")

# Label kelas
varieties = ['B11', 'CLYN', 'CY7','M1214', 'MAL03', '1026.12', 'N51', 'N79' ]
reclass_varieties = ['N51', 'N79']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(file.stream).convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            # Klasifikasi pertama (8 kelas)
            outputs = [F.softmax(model(input_tensor), dim=1) for model in models.values()]
            avg_probs = torch.mean(torch.stack(outputs), dim=0)  # Soft voting

            confidence, predicted = torch.max(avg_probs, 1)
            predicted_label = varieties[predicted.item()]
            confidence_percent = round(confidence.item() * 100, 2)
            confidence_per_class = {varieties[i]: round(avg_probs[0][i].item() * 100, 2) for i in range(len(varieties))}

            # Jika hasil awal N51 atau N79 → reclass menggunakan 2 kelas (soft voting)
            if predicted_label in ['N51', 'N79']:
                print(f"Reklasifikasi karena deteksi awal '{predicted_label}'")
                reclass_probs = [F.softmax(model(input_tensor), dim=1) for model in reclass_models.values()]
                reclass_avg = torch.mean(torch.stack(reclass_probs), dim=0)
                reclass_conf, reclass_pred = torch.max(reclass_avg, 1)
                predicted_label = reclass_varieties[reclass_pred.item()]
                confidence_percent = round(reclass_conf.item() * 100, 2)
                confidence_per_class = {reclass_varieties[i]: round(reclass_avg[0][i].item() * 100, 2) for i in range(len(reclass_varieties))}

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'variety': predicted_label,
            'confidence': confidence_percent,
            'all_confidences': confidence_per_class,
            'image_base64': img_str
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f"Kesalahan sistem: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
