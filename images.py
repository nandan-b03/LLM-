# Import required packages
import cv2
import pytesseract
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
vs = cv2.VideoCapture(0)

#read the image from camara
while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
                break

        # Convert the image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Performing OTSU threshold
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Creating a copy of image
        im2 = frame.copy()
        
        text = ''
        for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Drawing a rectangle on copied image
                rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Cropping the text block for giving input to OCR
                cropped = im2[y:y + h, x:x + w]
                
                # Apply OCR on the cropped image
                text += ' '+pytesseract.image_to_string(cropped)

        text=text.lower()
        print(text)
        
        cv2.imshow('Camara', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
vs.release()
cv2.destroyAllWindows()



if text.find('paracetamol') != -1:
        description = 'paracetamol or Dolo-650.'
elif text.find('duphalac') != -1:
        description = 'Duphalac. this is used for drinking purpose'
elif text.find('cissus quadrangularis tablets') != -1:
        description = 'write tablet description here'
elif text.find('tablet name') != -1:
        description = 'write tablet description here'
else:
        description ='result not found'

myobj = gTTS(text=description, lang='en', slow =False)
myobj.save("voice.mp3")
song=MP3('voice.mp3')
pygame.mixer.init()
pygame.mixer.music.load('voice.mp3')
pygame.mixer.music.play()
time.sleep(song.info.length)
pygame.quit()
