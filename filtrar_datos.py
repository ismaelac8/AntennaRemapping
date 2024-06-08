import numpy as np
import scipy.io

def filtrar_datos_en_rango(inputs, labels):
    inputs = np.array(inputs[0][0])
    labels = np.array(labels[0][0])

    # Crear un vector booleano con las condiciones especificadas
    validos = np.logical_and(inputs[:, 1] >= -25, inputs[:, 1] <= 25)

    # Filtrar inputs y labels según el vector booleano
    inputs_filtrados = inputs[validos, :]
    labels_filtrados = labels[:, :, validos]

    if len(inputs_filtrados) > 0:
        return inputs_filtrados, labels_filtrados
    else:
        print("No hay datos dentro del rango especificado.")
        return None, None