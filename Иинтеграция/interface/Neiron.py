import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

m_zero = tf.keras.models.load_model("Data_zero.h5")
m_first = tf.keras.models.load_model("Data_first.h5")
m_second = tf.keras.models.load_model("Data_second.h5")
m_zero_2 = tf.keras.models.load_model("Data_2_zero.h5")
m_first_2 = tf.keras.models.load_model("Data_2_first.h5")
m_second_2 = tf.keras.models.load_model("Data_2_second.h5")

def predict_models(massiv, massiv_2):

    #scaler = preprocessing.QuantileTransformer().fit(massiv)
    #scaler_2 = preprocessing.QuantileTransformer().fit(massiv_2)
    #massiv = scaler.transform(massiv)
    #massiv_2 = scaler_2.transform(massiv_2)
    massiv = np.reshape(massiv, (massiv.shape[0], 1, 240))
    massiv_2 = np.reshape(massiv_2, (massiv_2.shape[0], 1, 240))

    predict_zero = m_zero.predict(massiv)
    predict_first = m_first.predict(massiv)
    predict_second = m_second.predict(massiv)

    predict_zero_2 = m_zero_2.predict(massiv_2)
    predict_first_2 = m_first_2.predict(massiv_2)
    predict_second_2 = m_second_2.predict(massiv_2)

    foto = []
    piezo = []

    foto.append(mean_absolute_error(predict_zero[0][0], massiv[0][0]))
    foto.append(mean_absolute_error(predict_first[0][0], massiv[0][0]))
    foto.append(mean_absolute_error(predict_second[0][0], massiv[0][0]))

    piezo.append(mean_absolute_error(predict_zero_2[0][0], massiv_2[0][0]))
    piezo.append(mean_absolute_error(predict_first_2[0][0], massiv_2[0][0]))
    piezo.append(mean_absolute_error(predict_second_2[0][0], massiv_2[0][0]))

    class_foto = foto.index(min(foto))
    class_piezo = piezo.index(min(piezo))
    class_sred = (class_foto + class_piezo) / 2

    if float(class_sred) != int(class_sred):
        otvet = foto.index(min(foto))
    else:
        otvet = int(class_sred)

    mass = [class_foto,class_piezo,otvet]

    return mass