import easyocr
import cv2
import numpy as np

# Инициализация OCR с нужными языками
reader = easyocr.Reader(['en', 'ru'])

# Список изображений для обработки
images = ['/Users/khatabaev/Desktop/First image.jpg']

for img_path in images:
    # Загружаем изображение
    image = cv2.imread(img_path)
    
    # Преобразуем изображение в оттенки серого для улучшения контраста
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применяем бинаризацию для улучшения контраста
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    
    # Распознаем текст с предобработанным изображением
    result = reader.readtext(binary_image)
    
    # Проверяем, есть ли результаты
    if result:
        # Ищем результат с наибольшей уверенностью
        max_confidence_result = max(result, key=lambda x: x[2])
        
        # Выводим результат с наибольшей уверенностью
        print(f"Highest confidence result for {img_path}:")
        print(f"Detected text: {max_confidence_result[1]} (confidence: {max_confidence_result[2]:.2f})")
    else:
        print(f"No text detected in {img_path}.")
