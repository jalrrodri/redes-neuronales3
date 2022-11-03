import string
from gtts import gTTS

def crearaudio(texto):
    language = "es"
    myobj = gTTS(text=texto, lang=language, slow=False)
    texto = texto.replace(" ", "")
    myobj.save("audio/" + texto + ".mp3")

# def crearaudio_abecedario():
#     for i in string.ascii_uppercase:
#         language = "es"
#         myobj = gTTS(text=i, lang=language, slow=False)
#         myobj.save("audio/" + i + ".mp3")
