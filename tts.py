from gtts import gTTS

def crearaudio(texto):
    language = "es"
    myobj = gTTS(text=texto, lang=language, slow=False)
    texto = texto.replace(" ", "")
    myobj.save("audio/" + texto + ".mp3")
