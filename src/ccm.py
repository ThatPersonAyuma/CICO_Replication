import cv2
import numpy as np


def compress_jpeg(image_segment, quality):
    # input citra, nilainya bisa di rentang 1 sampai 100, output dekompresi
    quality = int(np.clip(quality, 1, 100))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # enkode ke bistream jpeg
    success, encoded_bytes = cv2.imencode(".jpg", image_segment, encode_params)
    if not success:
        raise ValueError("gagal enkode jpeg")

    payload_size = len(encoded_bytes)

    # dekode ke gambar
    decoded_segment = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)

    return decoded_segment, payload_size


def standar_jpeg(image, quality=50):
    # kompresi raw image
    decoded_img, payload_bytes = compress_jpeg(image, quality)

    print(f"standard uniform jpeg, q = {quality}")
    print(f"ukuran payload: {payload_bytes / 1024:.2f} KB\n")

    return decoded_img, payload_bytes
