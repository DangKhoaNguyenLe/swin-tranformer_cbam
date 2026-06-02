from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare(face_embedding, db_embedding):
    """
    So sánh độ tương đồng giữa hai vector embedding khuôn mặt.
    """
    raw_sim = cosine_similarity(face_embedding, db_embedding)[0][0]
    
    # Giá trị cosine similarity nằm trong khoảng [-1, 1], ta có thể trả về trực tiếp
    return float(raw_sim)