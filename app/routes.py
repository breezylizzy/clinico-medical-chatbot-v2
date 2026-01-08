from flask import Blueprint, render_template, request, jsonify, session

import io
import os
import json
import uuid
import numpy as np
from PIL import Image, ImageOps

from langchain_openai import ChatOpenAI

from app.config import Config
from src.chain import create_rag_chain

routes = Blueprint("routes", __name__)

# RULES (image classification)
LOW_CONF_THRESHOLD = 0.85
MAX_LOW_CONF_ATTEMPTS = 3
_low_conf_counter = {}  # session_id -> attempts (low-confidence streak)

# TensorFlow model (lazy load)
_tf_model = None

def get_model():
    global _tf_model
    if _tf_model is None:
        import tensorflow as tf
        _tf_model = tf.keras.models.load_model(Config.MODEL_PATH)
    return _tf_model

# Class mapping
with open(Config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_map = json.load(f)

IDX_TO_CLASS = {v: k for k, v in class_map.items()}

# RAG setup
chat_model = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.0
)

rag_chain = create_rag_chain(chat_model)

# Helpers
def get_session_id() -> str:
    """
    Ambil session_id dari client (FormData / header / query).
    Jika tidak ada, fallback ke Flask session (butuh app.secret_key).
    """
    sid = (
        request.form.get("session_id")
        or request.headers.get("X-Session-Id")
        or request.args.get("session_id")
    )
    if sid:
        return sid

    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess yang konsisten untuk inference:
    - EXIF transpose (foto HP)
    - RGB
    - resize 224x224
    - float32 / 255.0
    - batch dim
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


def format_prediction_line(label: str, conf_0_1: float) -> str:
    return f"Hasil klasifikasi gambar (bukan diagnosis): {label} — confidence {conf_0_1 * 100:.1f}%.\n"

# Routes pages
@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

# Main API
@routes.route("/chat", methods=["POST"])
def chat():
    """
    Response sukses:
      { ok:true, reply:"...", prediction:{disease,confidence}|null, attempt?:int }
    Response gagal:
      { ok:false, error:"..." }
    """
    try:
        user_text = (request.form.get("message") or "").strip()
        image_file = request.files.get("image")
        session_id = get_session_id()

        prediction_result = None
        pred_line = ""
        disease_context = ""

        # 1) IMAGE CLASSIFICATION (if any)
        if image_file:
            image_bytes = image_file.read()
            if not image_bytes:
                return jsonify({"ok": False, "error": "File gambar kosong."}), 400

            x = preprocess_image(image_bytes)
            tf_model = get_model()

            preds = tf_model.predict(x)[0]
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx])
            disease_label = IDX_TO_CLASS.get(class_idx, f"unknown_{class_idx}")

            prediction_result = {
                "disease": disease_label,
                "confidence": round(confidence, 3)
            }
            pred_line = format_prediction_line(disease_label, confidence)

            # 2) RULE: confidence < 90% => minta foto lebih jelas (max 3x)
            if confidence < LOW_CONF_THRESHOLD:
                attempts = _low_conf_counter.get(session_id, 0) + 1
                _low_conf_counter[session_id] = attempts

                if attempts < MAX_LOW_CONF_ATTEMPTS:
                    return jsonify({
                        "ok": True,
                        "reply": (
                            pred_line +
                            "Maaf, gambar yang Anda kirim belum cukup jelas untuk kami identifikasi dengan yakin.\n"
                            "Tolong unggah foto yang lebih jelas (pencahayaan terang, fokus tajam/tidak blur, "
                            "jarak lebih dekat, area kulit terlihat utuh)."
                        ),
                        "prediction": prediction_result,
                        "attempt": attempts
                    }), 200
                else:
                    return jsonify({
                        "ok": True,
                        "reply": (
                            pred_line +
                            "Maaf, kami tidak dapat mengidentifikasi kondisi kulit tersebut dari gambar yang Anda kirim.\n"
                            "Untuk keamanan, harap mengunjungi fasilitas kesehatan terdekat untuk mendapatkan bantuan langsung."
                        ),
                        "prediction": prediction_result,
                        "attempt": attempts
                    }), 200

            # 3) RULE: confidence >= 90% => reset attempt ke 0
            _low_conf_counter[session_id] = 0

            # Context untuk RAG hanya saat yakin
            disease_context = (
                f"Berdasarkan analisis gambar kulit yang diunggah, "
                f"terdapat indikasi penyakit kulit: {disease_label} "
                f"dengan tingkat keyakinan {confidence * 100:.1f}%."
            )

        # 4) BUILD QUERY KE RAG
        if user_text and disease_context:
            final_query = f"{disease_context}\n\nPertanyaan pengguna:\n{user_text}"
        elif user_text:
            final_query = user_text
        elif disease_context:
            final_query = (
                f"{disease_context}\n\n"
                "Tolong jelaskan kondisi tersebut secara umum dengan bahasa yang mudah dipahami oleh pasien."
            )
        else:
            return jsonify({"ok": False, "error": "Pesan atau gambar harus diisi."}), 400

        # 5) RAG CALL
        rag_response = rag_chain.invoke(
            {"input": final_query},
            config={"configurable": {"session_id": session_id}}
        )

        if isinstance(rag_response, dict):
            answer = rag_response.get("answer", "")
        else:
            answer = str(rag_response)

        # Kalau ada prediksi gambar, tampilkan di awal balasan
        final_reply = (pred_line + "\n" + answer) if prediction_result else answer

        return jsonify({
            "ok": True,
            "reply": final_reply,
            "prediction": prediction_result
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500