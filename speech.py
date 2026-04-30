
import pyaudio
import speech_recognition as sr
import pyttsx3
import datetime


l=sr.Recognizer()
engine=pyttsx3.init()
engine. setProperty('rate', 100)
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
engine.say('I am alexa, how can i help you')
engine.runAndWait()

def talk(text):
   engine.say(text)
   engine.runAndWait()

def take_command():
   try:
      with sr.Microphone() as source:
         print('listening.......')
         voice=l.listen(source)
         command=l.recognize_google(voice)
         command= command.lower()
         print(command)
   except:
      pass
   return command



def run_alexa():
   command = take_command()
   print(command)
   if 'time' in command:
      time=datetime.datetime.now().strftime('%I:%M %p')
      talk('ganta bandu'+ time)
      print(time)
   elif 'amaravati' in command:
      talk('she is the mother of sandeep and mother of meghana her husband is venkatesh. thats all about amaravathi')
   elif 'bye' in command:
      quit
      talk('good bye')
   else:
      talk('Please say the command again.')
   
while True:
   run_alexa()

