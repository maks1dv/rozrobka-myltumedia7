import numpy as np
from PIL import Image
from skimage import data

# ==========================================
# 2.4. Перетворення гами sRGB <-> Світло
# ==========================================

def to_light(c):
    """Перетворення коду sRGB (0..255) у лінійне світло (0..1)"""
    c_norm = c / 255.0
    return np.where(c_norm <= 0.04045, c_norm / 12.92, ((c_norm + 0.055) / 1.055) ** 2.4)

def to_code(L):
    """Перетворення лінійного світла (0..1) у код sRGB (0..255)"""
    c = np.where(L <= 0.0031308, L * 12.92, 1.055 * (L ** (1 / 2.4)) - 0.055)
    return np.clip(c * 255.0, 0, 255)

# ==========================================
# Функції зменшення вдвічі (2x2)
# ==========================================

def half_naive(img):
    """Наївне усереднення кодів sRGB 2x2"""
    h, w = img.shape
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

def half_linear(img):
    """Правильне усереднення у лінійному просторі світла 2x2"""
    L = to_light(img)
    h, w = L.shape
    return to_code(L.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)))

def checker(N=512, cell=4, ang=18):
    """Генерація шахівниці під кутом 18 градусів"""
    y, x = np.indices((N, N)).astype(float) - N / 2
    a = np.deg2rad(ang)
    xr = x * np.cos(a) - y * np.sin(a)
    yr = x * np.sin(a) + y * np.cos(a)
    return (((np.floor(xr / cell) + np.floor(yr / cell)) % 2) * 255.0)


# ==========================================
# Головний сценарій виконання
# ==========================================
if __name__ == "__main__":
    # --------------------------------------
    # Завдання 1. Збільшення
    # --------------------------------------
    print("=== Завдання 1. Збільшення ===")
    gray = np.array(Image.fromarray(data.astronaut()).convert('L'))
    small = gray[340:356, 372:388]  # Фрагмент 16x16
    im_small = Image.fromarray(small)

    im_small.resize((320, 320), Image.NEAREST).save('up_nearest.png')
    im_small.resize((320, 320), Image.BILINEAR).save('up_bilinear.png')
    im_small.resize((320, 320), Image.LANCZOS).save('up_lanczos.png')
    print("Файли up_nearest.png, up_bilinear.png, up_lanczos.png успішно збережено.\n")

    # --------------------------------------
    # Завдання 2. Зменшення й аліасинг
    # --------------------------------------
    print("=== Завдання 2. Зменшення й аліасинг ===")
    ch = checker()
    k = 4
    nearest = ch[::k, ::k]  # Кожен 4-й піксель
    area = ch.reshape(512 // k, k, 512 // k, k).mean(axis=(1, 3))  # Усереднення 4x4

    for name, im in (('ds_nearest', nearest), ('ds_area', area)):
        Image.fromarray(im.astype(np.uint8)).save(name + '.png')
        print(f"{name:10s} | Середнє: {im.mean():5.1f} | Розкид (std): {im.std():5.1f}")
    print()

    # --------------------------------------
    # Завдання 3. Ланцюжок міпмап (наївний)
    # --------------------------------------
    print("=== Завдання 3. Ланцюжок міпмап ===")
    g = gray.astype(float)
    level = 0
    while g.shape[0] >= 2:
        Image.fromarray(g.astype(np.uint8)).save(f'mip_{level}.png')
        print(f"Level {level:2d} | Розмір: {g.shape[0]:3d}x{g.shape[1]:3d} | Середня яскравість: {g.mean():.1f}")
        g = half_naive(g)
        level += 1
    print()

    # --------------------------------------
    # Завдання 4. Вплив гами
    # --------------------------------------
    print("=== Завдання 4. Тест на 1-піксельній шахівниці ===")
    cb = np.indices((256, 256)).sum(axis=0) % 2 * 255.0
    print(f"Наївно:    {half_naive(cb).mean():.1f}")
    print(f"Правильно: {half_linear(cb).mean():.1f}\n")

    print("=== Завдання 4. Накопичення різниці на фотографії ===")
    gn = gray.astype(float)
    gl = gray.astype(float)
    
    print(f"Рівень 0 | Розмір: {gn.shape[0]:3d} | Наївно: {gn.mean():.1f} | Правильно: {gl.mean():.1f} | Різниця: {gl.mean() - gn.mean():+.1f}")
    for lv in range(1, 6):
        gn, gl = half_naive(gn), half_linear(gl)
        diff = gl.mean() - gn.mean()
        print(f"Рівень {lv} | Розмір: {gn.shape[0]:3d} | Наївно: {gn.mean():.1f} | Правильно: {gl.mean():.1f} | Різниця: {diff:+.1f}")

    # --------------------------------------
    # Бонус перевірка Pillow
    # --------------------------------------
    print("\n=== Додаткова перевірка Pillow ===")
    im_cb = Image.fromarray(cb.astype(np.uint8))
    box_res = np.array(im_cb.resize((128, 128), Image.BOX)).mean()
    print(f"Pillow BOX режим на 1-піксельній шахівниці дає: {box_res:.1f} (гама не враховується!)")