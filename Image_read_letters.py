import easyocr as ea

reader = ea.Reader(['en', 'ru', 'uz'])

images = ['/Users/khatabaev/Desktop/First image.jpg']

result = reader.readtext(images)

print(result)
