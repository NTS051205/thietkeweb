from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocess import preprocess_text, load_tokenizer

# Tải mô hình LSTM
lstm_model = load_model("saved_model/saved_models/lstm_model.h5")

# Tải tokenizer
tokenizer = load_tokenizer("saved_model/tokenizer.pkl")  # Đảm bảo bạn đã lưu tokenizer dưới dạng .pkl

def interpret_result(prediction):
    """
    Chuyển đổi kết quả dự đoán thành nhãn cảm xúc.
    :param prediction: Mảng kết quả dự đoán từ mô hình.
    :return: Nhãn cảm xúc (vd: Positive, Negative, Neutral).
    """
    classes = ["Negative", "Neutral", "Positive"]  # Tuỳ chỉnh theo mô hình của bạn
    return classes[prediction.argmax()]

def predict_lstm(text):
    """
    Hàm dự đoán cảm xúc từ văn bản bằng mô hình LSTM.
    :param text: Văn bản cần phân tích.
    :return: Kết quả phân tích cảm xúc.
    """
    # Tiền xử lý văn bản
    processed_text = preprocess_text(text)

    # Chuyển đổi văn bản thành chuỗi số và padding
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # maxlen phải khớp với mô hình

    # Dự đoán
    prediction = lstm_model.predict(padded_sequence)

    # Chuyển đổi kết quả thành nhãn cảm xúc
    return interpret_result(prediction)
