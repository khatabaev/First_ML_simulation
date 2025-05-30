import easyocr
import cv2
import numpy as np
import os

# Список путей к изображениям
image_paths = [
    '/Users/khatabaev/Desktop/Se i.png',
    '/Users/khatabaev/Desktop/Uzb_image.png',
    # '/Users/khatabaev/Desktop/image2.png',
    # '/Users/khatabaev/Desktop/image3.jpg',
]

# Минимальный порог уверенности (49%)
MIN_CONFIDENCE = 0.49

# Инициализируем OCR один раз для оптимизации
print("Инициализация OCR...")
reader_ru = easyocr.Reader(['en', 'ru'])
reader_uz = easyocr.Reader(['en', 'uz'])

# Обрабатываем каждое изображение
for image_path in image_paths:
    print(f"\n{'='*50}")
    print(f"Обработка изображения: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    
    # Читаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось прочитать изображение по пути {image_path}")
        continue  # Переходим к следующему изображению

    # Ищем русский текст
    print("\n=== Русский текст ===")
    results_ru = reader_ru.readtext(image)
    found_ru = False
    for (bbox, text, prob) in results_ru:
        if prob >= MIN_CONFIDENCE:
            found_ru = True
            print(f"Найден текст: {text} (уверенность: {prob:.2f})")
    
    if not found_ru:
        print("Русский текст не найден или уверенность ниже порога")
    
    # Ищем узбекский текст
    print("\n=== Узбекский текст ===")
    results_uz = reader_uz.readtext(image)
    found_uz = False
    for (bbox, text, prob) in results_uz:
        if prob >= MIN_CONFIDENCE:
            found_uz = True
            print(f"Найден текст: {text} (уверенность: {prob:.2f})")
    
    if not found_uz:
        print("Узбекский текст не найден или уверенность ниже порога")
    
    print(f"\nПоказаны только результаты с уверенностью выше {MIN_CONFIDENCE*100}%")

