import easyocr as ea

reader = ea.Reader(['en', 'ru', 'uz_latn'])

images = ['/Users/khatabaev/Desktop/First image.jpg']

result = reader.readtext(images)

print(result)
