import easyocr

# Инициализация с нужными языками
reader = easyocr.Reader(['en', 'ru', 'uz_latn'])

images = ['/Users/khatabaev/Desktop/First image.jpg']

for img_path in images:
    result = reader.readtext(img_path)
    print(f"Results for {img_path}:")
    for detection in result:
        print(detection[1])  # распознанный текст
