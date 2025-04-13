from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from utils.preprocess import preprocess_text

# Tải tokenizer và mô hình PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert_model = TFAutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base", num_labels=3
)

def interpret_result(probabilities):
    """
    Chuyển đổi kết quả dự đoán thành nhãn cảm xúc.
    :param probabilities: Xác suất dự đoán từ mô hình (logits đã qua softmax).
    :return: Nhãn cảm xúc (vd: Positive, Negative, Neutral).
    """
    classes = ["Negative", "Neutral", "Positive"]  # Thứ tự phải khớp với huấn luyện
    predicted_label = tf.argmax(probabilities, axis=-1).numpy()[0]
    return classes[predicted_label]

def predict_phobert(text):
    """
    Hàm dự đoán cảm xúc từ văn bản bằng PhoBERT.
    :param text: Văn bản cần phân tích.
    :return: Kết quả phân tích cảm xúc.
    """
    # Tiền xử lý văn bản (nếu cần)
    processed_text = preprocess_text(text)

    # Tokenize
    inputs = tokenizer(
        processed_text, return_tensors="tf", truncation=True, padding=True, max_length=256
    )

    # Dự đoán với mô hình PhoBERT
    outputs = phobert_model(inputs)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)

    # Chuyển đổi kết quả thành nhãn cảm xúc
    return interpret_result(probabilities)
