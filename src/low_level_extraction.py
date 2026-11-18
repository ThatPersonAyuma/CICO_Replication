import cv2
import matplotlib.pyplot as plt


def plot_keypoints(img_gray, keypoints_dict, save_path=None):
    """Memvisualisasikan keypoint dari berbagai detector secara berdampingan."""
    # Konversi gambar grayscale ke BGR agar titik berwarna (hijau) terlihat jelas
    img_base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    jumlah_plot = len(keypoints_dict)
    fig, axes = plt.subplots(1, jumlah_plot, figsize=(6 * jumlah_plot, 6))

    if jumlah_plot == 1:
        axes = [axes]

    for ax, (nama_algo, kp) in zip(axes, keypoints_dict.items()):
        # Menggambar keypoint beserta ukuran skala dan orientasinya
        img_kp = cv2.drawKeypoints(
            img_base,
            kp,
            None,
            color=(0, 255, 0),  # Warna hijau (format BGR)
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        # Konversi BGR (OpenCV) ke RGB (Matplotlib) sebelum ditampilkan
        img_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.set_title(f"{nama_algo}\n({len(kp)} keypoints)", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

SAMPLE_PATH = "sample2.jpg"

# ==========================================
# Contoh Penggunaan
# ==========================================
img = cv2.imread("sample2.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Deteksi dengan FAST
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
kp_fast = fast.detect(img, None)

# 2. Deteksi dengan STAR (membutuhkan opencv-contrib-python)
star = cv2.xfeatures2d.StarDetector_create()
kp_star = star.detect(img, None)

# 3. Deteksi dengan ORB (sebagai pembanding)
orb = cv2.ORB_create(nfeatures=500)
kp_orb = orb.detect(img, None)

# Panggil fungsi plotting
hasil_deteksi = {
    "FAST (Corner)": kp_fast,
    "STAR (Blob)": kp_star,
    "ORB (Oriented Corner)": kp_orb,
}

plot_keypoints(img, hasil_deteksi, save_path=f"perbandingan_keypoints_{SAMPLE_PATH.split("/")[-1].split(".")[0]}.png")