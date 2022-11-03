#----------------------------------------- Se importan las librerias ------------------------------------------
import cv2
import Informacion_manos as im
from playsound import playsound
import os
import  numpy as np
from keras_preprocessing.image import  load_img, img_to_array
from tensorflow.python.keras.models import load_model
import time
from threading import Timer

#-------------------------------------------------------------Funcion para decir el objeto en voz alta------------------------------------------------------------
def voz(nombre_objeto):
    return playsound(u"audio/" + nombre_objeto + ".mp3")
# una_vez = 0
# while 1:
#     if una_vez == 0:
#         voz()
#         una_vez = 1

#------------------------------------------------------ Ubicacion del modelo y los pesos ---------------------------------------------------------
modelo = 'tmp\saved_model\SavedModel.h5'
peso = 'tmp\saved_model\SavedWeights.h5'

#----------------------------------------------------------- Se carga el modelo -----------------------------------------------------------------
cnn = load_model(modelo) #Se carga el modelo
cnn.load_weights(peso) #Se carga los pesos

#----------------------------------------------------------- Se carga los nombres de las carpetas -----------------------------------------------
direccion = modelo = 'Fotos\Entrenamiento'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

#------------------------------------------------------------------ Se declaran las variables ---------------------------------------------------
anchocam, altocam = 640, 480


#----------------------------------------------------------------------Lectura de la camara -----------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, anchocam) #Se define un ancho y un alto definido para siempre
cap.set(4, altocam)

#-------------------------------------------------------------Se declara el detector ------------------------------------------------------------
detector = im.detectormanos(maxManos=1, Confdeteccion=0.7) #Solo se utilizara una mano

while True:
    #------------------------Se va a encontrar los puntos de la mano ----------------------------------------------------------------
    ret, frame = cap.read()
    mano = detector.encontrarmanos(frame) #Encontramos las manos
    lista, bbox = detector.encontrarposicion(frame) #Se muestran las posiciones
    if len(lista) != 0:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        data = frame[y1:y2, x1:x2]
        try:
            obje = cv2.resize(data, (200, 200), interpolation = cv2.INTER_CUBIC) #Redimensiones de las fotos
        except Exception as e:
            print(e)
        x = img_to_array(obje) # Se convierte la imagen a una matriz
        x = np.expand_dims(x, axis=0) #Se agrega un nuevo eje
        vector = cnn.predict(x) # Sera un arreglo de 2 dimensiones, donde se va a poner 1 en la clase que crea correcta
        resultado = vector[0] # [1,0,0] [0,1,0] [0,0,1]
        respuesta = np.argmax(resultado) #Nos entrega el indice del valor más alto
        if respuesta == 0:
            print(resultado)
            #cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[0]))
            # t = Timer(5, voz('{}'.format(dire_img[0])))
            # t.start()
        elif respuesta == 1:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 2:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(dire_img[2]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 3:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[3]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 4:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[4]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 5:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[5]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 6:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[6]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 7:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[7]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 8:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[8]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 9:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[9]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 10:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[10]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 11:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[11]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 12:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[12]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 13:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[13]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 14:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[14]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 15:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[15]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 16:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[16]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 17:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[17]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 18:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[18]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 19:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[19]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 20:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[20]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 21:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[21]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 22:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[22]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 23:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[23]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 24:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[24]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 25:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[25]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        elif respuesta == 26:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[26]), (x1, y1 - 5), 1, 2.5, (0, 0, 255), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[2]))
            # t = Timer(5, voz('{}'.format(dire_img[2])))
            # t.start()
        elif respuesta == 27:
            print(resultado)
            # cv2.rectangle(frame,(x1, y1), (x2, y2), (0,255, 0), 3)
            cv2.putText(frame, '{}'.format(
                dire_img[27]), (x1, y1 - 5), 1, 2.5, (0, 255, 0), 3, cv2.LINE_AA)
            # voz('{}'.format(dire_img[1]))
            # t = Timer(5, voz('{}'.format(dire_img[1])))
            # t.start()
        else:
            cv2.putText(frame, 'OBJETO DESCONOCIDO', (x1, y1 - 5), 1,1.3, (0, 0, 0), 1, cv2.LINE_AA)
            # voz("Objetodesconocido.mp3")
            # t = Timer(5, voz("Objetodesconocido.mp3"))
            # t.start()

    cv2.imshow("Clasificador", frame)
    k = cv2.waitKey(1)
    if k ==  27:
        break
cap.release()
cv2.destroyAllWindows()