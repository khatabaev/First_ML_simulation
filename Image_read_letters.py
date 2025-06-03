import easyocr
import cv2
import numpy as np
import os

# Список путей к изображениям
image_paths = [
    '/Users/khatabaev/Desktop/Image from Igor 1.jpg',
    '/Users/khatabaev/Desktop/Image from Igor 2.jpg',
    '/Users/khatabaev/Desktop/Image from Igor 3.jpg',
    '/Users/khatabaev/Desktop/Image from Igor 4.jpg'
]

# Минимальный порог уверенности, чтоб фиговые результаты модель не показывала
MIN_CONFIDENCE = 0.49

def save_processed_images(original, gray, thresh, denoised, base_filename):
    # Создаем папку processed_images на рабочем столе, если её нет
    output_dir = '/Users/khatabaev/Desktop/processed_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Получаем имя файла без расширения
    filename = os.path.splitext(os.path.basename(base_filename))[0]
    
    # Сохраняем каждый этап обработки
    cv2.imwrite(os.path.join(output_dir, f"{filename}_original.jpg"), original)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_gray.jpg"), gray)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_threshold.jpg"), thresh)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_denoised.jpg"), denoised)

def preprocess_image(image, save_path=None):
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применение адаптивного порога для улучшения контраста
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Удаление шума
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    # Если указан путь для сохранения, сохраняем все этапы обработки
    if save_path:
        save_processed_images(image, gray, thresh, denoised, save_path)
    
    return denoised

# Инициализируем OCR один раз для оптимизации
print("Инициализация OCR...")
reader_ru = easyocr.Reader(['en', 'ru'])
reader_uz = easyocr.Reader(['en', 'uz'])

def determine_language(text): # для того, чтобы опередлить язык и ограничить количество распознаваний на схожие языки
    ru_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    uz_chars = set('ўқғҳ')
    en_chars = set('abcdefghijklmnopqrstuvwxyz')
    
    text = text.lower()
    ru_count = sum(1 for char in text if char in ru_chars)
    uz_count = sum(1 for char in text if char in uz_chars)
    en_count = sum(1 for char in text if char in en_chars)
    
    if uz_count > 0:
        return 'uz'
    elif ru_count > 0:
        return 'ru'
    elif en_count > 0:
        return 'en'
    return 'unknown'

# Обрабатываем каждое изображение
for image_path in image_paths:
    print(f"\n{'='*50}")
    print(f"Обработка изображения: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    
    # Читаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: Не удалось прочитать изображение по пути {image_path}")
        continue

    # Предобработка изображения с сохранением промежуточных результатов
    processed_image = preprocess_image(image, save_path=image_path)
    print(f"Промежуточные результаты обработки сохранены в папку processed_images на рабочем столе")
    
    # Пробуем оба варианта изображения - оригинал и обработанное
    all_results = []
    
    # Проверяем оригинальное изображение
    results_ru_orig = reader_ru.readtext(image)
    results_uz_orig = reader_uz.readtext(image)
    
    # Проверяем обработанное изображение
    results_ru_proc = reader_ru.readtext(processed_image)
    results_uz_proc = reader_uz.readtext(processed_image)
    
    # Объединяем все результаты
    all_results = results_ru_orig + results_uz_orig + results_ru_proc + results_uz_proc
    
    # Удаляем дубликаты и сортируем по уверенности
    seen_texts = set()
    unique_results = []
    
    for (bbox, text, prob) in all_results:
        if text.lower() not in seen_texts and prob >= MIN_CONFIDENCE:
            seen_texts.add(text.lower())
            unique_results.append((bbox, text, prob))
    
    # Сортируем по уверенности
    unique_results.sort(key=lambda x: x[2], reverse=True)
    
    # Выводим результаты по языкам
    ru_en_found = False
    uz_en_found = False
    
    for (bbox, text, prob) in unique_results:
        lang = determine_language(text)
        if lang in ['ru', 'en']:
            if not ru_en_found:
                print("\n=== Русский и английский текст ===")
                ru_en_found = True
            print(f"Найден текст: {text} (уверенность: {prob:.2f})")
        elif lang in ['uz', 'en']:
            if not uz_en_found:
                print("\n=== Узбекский и английский текст ===")
                uz_en_found = True
            print(f"Найден текст: {text} (уверенность: {prob:.2f})")
    
    if not ru_en_found and not uz_en_found:
        print("\nТекст не распознан как русский, английский или узбекский")
    
    print(f"\nПоказаны только результаты с уверенностью выше {MIN_CONFIDENCE*100}%")

