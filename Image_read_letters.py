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
    
    # Выводим результаты
    print(f"Results for {img_path}:")
    
    if result:
        for detection in result:
            # detection[1] — это текст, распознанный с картинки
            print(f"Detected text: {detection[1]} (confidence: {detection[2]:.2f})")
    else:
        print("No text detected.")
