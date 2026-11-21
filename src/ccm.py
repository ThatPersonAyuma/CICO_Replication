import cv2
import numpy as np

# def compress_jpeg(image_segment, quality):
#     # input citra, nilainya bisa di rentang 1 sampai 100, output dekompresi
#     quality = int(np.clip(quality, 1, 100))
#     encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

#     # enkode ke bistream jpeg
#     success, encoded_bytes = cv2.imencode(".jpg", image_segment, encode_params)
#     if not success:
#         raise ValueError("gagal enkode jpeg")

#     payload_size = len(encoded_bytes)

#     # dekode ke gambar
#     decoded_segment = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)

#     return decoded_segment, payload_size


# def standar_jpeg(image, quality=50):
#     # kompresi raw image
#     decoded_img, payload_bytes = compress_jpeg(image, quality)

#     print(f"standard uniform jpeg, q = {quality}")
#     print(f"ukuran payload: {payload_bytes / 1024:.2f} KB\n")

#     return decoded_img, payload_bytes


def split_image_into_tiles(image, grid_rows=4, grid_cols=4):
    # bagi gambar ke N petak, tepatnya r x c
    height, width, _ = image.shape
    tile_h = height // grid_rows
    tile_w = width // grid_cols
    tiles = []

    for row in range(grid_rows):
        row_start = row * tile_h
        row_end = (row + 1) * tile_h

        for column in range(grid_cols):
            column_start = column * tile_w
            column_end = (column + 1) * tile_w
            tile = image[row_start:row_end, column_start:column_end]
            tiles.append(tile)

    return tiles, (tile_h, tile_w), (grid_rows, grid_cols)


def stitch_tiles(compressed_tiles, tile_size, grid_shape):
    # merge gambar dari tahap awal jadi satu-kesatuan
    tile_h, tile_w = tile_size
    grid_rows, grid_cols = grid_shape
    full_h = tile_h * grid_rows
    full_w = tile_w * grid_cols

    # Create an empty image large enough to contain every tile.
    reconstructed_image = np.zeros(
        (full_h, full_w, 3),
        dtype=np.uint8,
    )

    tile_index = 0

    # Place each tile in its corresponding row and column.
    for row in range(grid_rows):
        row_start = row * tile_h
        row_end = row_start + tile_h

        for column in range(grid_cols):
            column_start = column * tile_w
            column_end = column_start + tile_w

            reconstructed_image[
                row_start:row_end,
                column_start:column_end,
            ] = compressed_tiles[tile_index]

            tile_index += 1

    return reconstructed_image


def mock_features(num_features=3, num_tiles=16):
    """Create random feature densities with each feature summing to one.

    The returned array has one row per feature and one column per tile.
    Each row is normalized independently, so it can be interpreted as a
    distribution of a feature across all tiles.
    """
    random_densities = np.random.rand(num_features, num_tiles)
    row_totals = random_densities.sum(axis=1, keepdims=True)
    normalized_densities = random_densities / row_totals

    return normalized_densities


def compute_qual(d, num_features=3, num_tiles=16):
    # Generate random feature weights and normalize them so they sum to 1.
    feature_weights = np.random.rand(num_features)
    feature_weights = feature_weights / feature_weights.sum()

    # Combine all feature densities for each tile using a weighted sum.
    tile_scores = np.dot(feature_weights, d)

    # Apply a random exponent to reduce or increase score differences.
    beta = np.random.uniform(low=0.5, high=2.0)

    maximum_score = np.max(tile_scores)
    if maximum_score == 0:
        normalized_scores = np.zeros_like(tile_scores)
    else:
        normalized_scores = tile_scores / maximum_score

    nonlinear_scores = np.power(normalized_scores, beta)

    # Map normalized scores from the range [0, 1] to JPEG quality [15, 90].
    minimum_quality = 15
    maximum_quality = 90

    quality_values = minimum_quality + (
        nonlinear_scores * (maximum_quality - minimum_quality)
    )

    quality_values = np.round(quality_values)
    jpeg_qualities = np.clip(
        quality_values,
        minimum_quality,
        maximum_quality,
    ).astype(int)

    return jpeg_qualities


def compress_tile_jpeg(tile, quality):
    """Compress one image tile with JPEG and decode it again."""
    jpeg_quality = int(quality)
    jpeg_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    encoding_succeeded, encoded_jpeg = cv2.imencode(
        ".jpg",
        tile,
        jpeg_parameters,
    )

    if not encoding_succeeded:
        raise ValueError("Failed to encode the image tile as JPEG.")

    # Decode the JPEG data to simulate the image received by the server.
    decoded_tile = cv2.imdecode(encoded_jpeg, cv2.IMREAD_COLOR)
    encoded_size_in_bytes = len(encoded_jpeg)

    return decoded_tile, encoded_size_in_bytes


def piperun(image_path):
    """Compress an image tile by tile and reconstruct the result."""
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Gambar tidak ditemukan")

    grid_rows = 4
    grid_cols = 4
    num_features = 3
    num_tiles = grid_rows * grid_cols

    # Divide the image into tiles.
    tiles, tile_size, grid_shape = split_image_into_tiles(
        img,
        grid_rows,
        grid_cols,
    )

    # Generate feature data and calculate one JPEG quality for each tile.
    d_mock = mock_features(num_features, num_tiles)
    qualities = compute_qual(d_mock, num_features, num_tiles)

    print(f"kualitas jpeg tiap petak: N:{num_tiles} : {qualities}")

    # Compress every tile and keep track of the total encoded size.
    compressed_tiles = []
    total_size_bytes = 0

    for tile, quality in zip(tiles, qualities):
        decoded_tile, encoded_size = compress_tile_jpeg(tile, quality)
        compressed_tiles.append(decoded_tile)
        total_size_bytes += encoded_size

    # Place the compressed tiles back into their original grid.
    output_img = stitch_tiles(compressed_tiles, tile_size, grid_shape)
    total_size_kb = total_size_bytes / 1024
    print(f"total ukuran gambar setelah kompres: {total_size_kb: .2f} KB")

    return output_img


def sample_test_piperun():
    """Run the compression pipeline with a temporary dummy image."""
    import os
    import tempfile

    dummy_image = np.random.randint(
        0,
        256,
        size=(128, 128, 3),
        dtype=np.uint8,
    )

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_file:
        image_path = image_file.name

    try:
        if not cv2.imwrite(image_path, dummy_image):
            raise AssertionError("Failed to write the dummy image.")

        result = piperun(image_path)

        assert result.shape == dummy_image.shape
        assert result.dtype == np.uint8
        return result
    finally:
        os.remove(image_path)


if __name__ == "__main__":
    sample_test_piperun()
