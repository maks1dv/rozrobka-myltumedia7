import time
import cv2
import numpy as np
from PIL import Image
from skimage import data

def conv2(img: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Двовимірна згортка з обробкою країв mode='edge'."""
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    out = np.zeros_like(img, dtype=float)
    for i in range(kh):
        for j in range(kw):
            out += K[i, j] * pad[i:i + img.shape[0], j:j + img.shape[1]]
    return out

def gauss1d(sigma: float, n: int) -> np.ndarray:
    """Генерує та нормує 1D ядро Гауса."""
    x = np.arange(n) - n // 2
    k = np.exp(-x**2 / (2 * sigma**2))
    return k / k.sum()

def conv_sep(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Роздільна (одномірна) згортка у два проходи."""
    n = len(k) // 2
    
    # Горизонтальний прохід
    pad = np.pad(img, ((0, 0), (n, n)), mode='edge')
    tmp = np.zeros_like(img, dtype=float)
    for j, v in enumerate(k):
        tmp += v * pad[:, j:j + img.shape[1]]
        
    # Вертикальний прохід
    pad = np.pad(tmp, ((n, n), (0, 0)), mode='edge')
    out = np.zeros_like(img, dtype=float)
    for i, v in enumerate(k):
        out += v * pad[i:i + img.shape[0], :]
        
    return out

def best(f, r=5) -> float:
    """Замір найкращого часу виконання у мс."""
    ts = []
    for _ in range(r):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000.0

if __name__ == '__main__':
    # Завантаження зображення Astronaut (512x512)
    gray = np.array(Image.fromarray(data.astronaut()).convert('L')).astype(float)
    
    # 1. Фільтри
    k_blur = gauss1d(1.6, 7)
    BLUR_KERNEL = np.outer(k_blur, k_blur)
    blur_res = conv2(gray, BLUR_KERNEL)
    
    SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    SY = SX.T
    gx, gy = conv2(gray, SX), conv2(gray, SY)
    edges = np.hypot(gx, gy)
    edges_scaled = edges / edges.max() * 255.0

    detail = gray - blur_res
    sharp_res = np.clip(gray + 1.2 * detail, 0, 255)

    # Збереження зображень
    Image.fromarray(blur_res.astype(np.uint8)).save('blur.png')
    Image.fromarray(edges_scaled.astype(np.uint8)).save('sobel.png')
    Image.fromarray(sharp_res.astype(np.uint8)).save('sharp.png')
    print("[+] Картинки blur.png, sobel.png, sharp.png успішно створені.")

    # 2. Заміри продуктивності
    print("\nТаблиця замірів часу (512x512):")
    print(f"{'Ядро':<8} | {'Звичайне (мс)':<14} | {'Роздільне (мс)':<14} | {'Виграш':<8} | {'OpenCV (мс)':<10}")
    print("-" * 65)
    for n in (3, 5, 9, 15, 31):
        sigma = max(n / 5.0, 0.6)
        k_vec = gauss1d(sigma, n)
        K_2d = np.outer(k_vec, k_vec)
        
        ta = best(lambda: conv2(gray, K_2d))
        tb = best(lambda: conv_sep(gray, k_vec))
        tc = best(lambda: cv2.GaussianBlur(gray, (n, n), sigma, borderType=cv2.BORDER_REPLICATE))
        print(f"{n}x{n:<5} | {ta:<14.2f} | {tb:<14.2f} | {ta/tb:<8.1f}x | {tc:<10.2f}")

    # 3. Перевірка розбіжності з OpenCV
    n_cv, sigma_cv = 15, 3.0
    k_cv = gauss1d(sigma_cv, n_cv)
    mine_cv = conv_sep(gray, k_cv)
    lib_cv = cv2.GaussianBlur(gray, (n_cv, n_cv), sigma_cv, borderType=cv2.BORDER_REPLICATE)
    print(f"\nРозбіжність з OpenCV: {np.abs(mine_cv - lib_cv).max():.2e}")
