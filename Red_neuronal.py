#-----------------------------------Se importa librerias -------------------------------
import keras.optimizers

#----------------------------------Crear modelo y entrenarlo ---------------------------
# Nos ayuda a preprocesar las imagenes que se le entregan
from keras_preprocessing.image import ImageDataGenerator
from keras import optimizers #Optimizadpr con el que se va a entrenar el witle
from keras.models import Sequential #Nos permite hacer redes neuronales secuenciales
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D #Capas para hacer les convoluciones
from keras import backend as K #Si hay una sesion de keras lo cerramos para tener todo Limpio
import os

K.clear_session() #Limpiamos todo

datos_entrenamiento = 'Fotos/Entrenamiento'
datos_validacion = 'Fotos/Validacion'

#Parametros
iteraciones = 20 #Numere de iteraciones pare alustalihowstro modele
altura, longitud = 200, 200 #Tamaño de las imagenes de entrenamiento
batch_size = 1 #Numero de imagenes que vamos a enviar
pasos = 300/1 #numero de veces que se va a procesar la informacion en casa iteracion
pasos_validacion = 300/1  # Despues de cada iteracion, validamos lo anterior
filtrosconv1 = 32
filtrosconv2 = 64   #Numero de filtros que vamos a aplicar en cada convolucion
filtrosconv3 = 128
tam_filtro1 = (4, 4)
tam_filtro2 = (3, 3)  # Tamaños de los filtros 1 2 y 3
tam_filtro3 = (2,2)
tam_pool = (2,2) #Tamaño del filtro en max pooling
clases = len(next(os.walk('Fotos\Entrenamiento'))[1]) # retorna el numero de clases basado en el numero de carpetas en el directorio

lr = 0.0005 #Ajustes de la red neuronal para acercarse a una solución optima

#Pre-Procesamiento de las imagenes
preprocesamiento_entre = ImageDataGenerator(
    rescale = 1./255, #Pasar los pixeles de 0 a 255 | 0 a 1
    shear_range = 0.3, #Generar nuestras imagenes inclinadas para un mejor entrenamiento
    zoom_range = 0.3, #Genera imagenes con zoom para un mejor entrenamiento
    horizontal_flip = True #Invierte las imagenes para mejor entrenamiento
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1./255
)

imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,  #Va a tomar las fotos que ya almacenamos
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical' #Clasificación categorica = por clases
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

#Se crea la red neuronal convolucional [CNN]
cnn = Sequential() #Red neuronal secuencial
#Se agrega filtros con el fin de volver la imagen muy profunda pero pequeña
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding = 'same', input_shape = (altura, longitud, 3), activation = 'relu'))
            #Es una convolucion y realizamos config
cnn.add(MaxPooling2D(pool_size=tam_pool)) #Despues de la primera capa vamos a tener una capa de max pooling y asignamos el tamaño
                                          #Maxpooling es la extraccion de caracteristicas

cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding = 'same', activation='relu'))# Agregamos nueva capa
cnn.add(MaxPooling2D(pool_size=tam_pool))

#Nueva capa
cnn.add(Convolution2D(filtrosconv3, tam_filtro3, padding = 'same', activation='relu'))# Agregamos nueva capa
cnn.add(MaxPooling2D(pool_size=tam_pool))

#Ahora se va a convertir esa imagen profunda a una plana, para tener 1 dimension con toda la info
cnn.add(Flatten()) #Aplanamos la imagen
cnn.add(Dense(384, activation='relu')) #Se asigna 426 neuronas
cnn.add(Dropout(0.5)) #Se apaga el 50% de las neuronas en la funcion enterior para no sobreajustar la red
cnn.add(Dense(clases, activation='softmax')) #Es nuestra ultima capa, es la que nos dice la probabilidad de que sea alguna

#Agregamos parametros para optimizar el modelo
#Durante el entrentamiento una autoevaluacion
optimizar = keras.optimizers.Adam(learning_rate= lr)
cnn.compile(loss = 'categorical_crossentropy', optimizer = optimizar, metrics = ['accuracy'])

#Se entrenara la red
cnn.fit(imagen_entreno, steps_per_epoch = pasos, epochs = iteraciones, validation_data = imagen_validacion,validation_steps = pasos_validacion )

#Guardamos el modelo
cnn.save('tmp/saved_model/SavedModel.h5')
cnn.save_weights('tmp/saved_model/SavedWeights.h5')
