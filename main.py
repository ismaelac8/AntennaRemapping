import numpy as np
import scipy
import scipy.io
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from filtrar_datos import filtrar_datos_en_rango
from procesar_tableros import procesar_tableros
from sklearn.preprocessing import StandardScaler

# Definir función objetivo para Optuna con K-Fold Cross Validation
def objective(trial):
    num_centers = trial.suggest_int("num_centers", 2, len(rs_tableros_procesados_1b) // 2)
    sigma = trial.suggest_float("sigma", 0.1, 1.0)
    lambda_reg = trial.suggest_float("lambda_reg", 0.01, 1.0)
            
    kf = KFold(n_splits=5)
    mse_scores = []
    for train_index, test_index in kf.split(rs_tableros_procesados_1b):
        X_train1, X_test1 = rs_tableros_procesados_1b[train_index], rs_tableros_procesados_1b[test_index]
        y_train1, y_test1 = y_labels[train_index], y_labels[test_index]
                
        rbf_net = RBFNetwork(num_centers=num_centers, sigma=sigma)
        rbf_net.fit(X_train1, y_train1, lambda_reg=lambda_reg)
        predictions_test = rbf_net.predict(X_test1)
        mse_scores.append(mean_squared_error(y_test1, predictions_test))
            
    return np.mean(mse_scores)

class RBFNetwork:
    def __init__(self, num_centers, sigma):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _calculate_activations(self, X):
        distances = scipy.spatial.distance.cdist(X, self.centers)
        activations = np.exp(-distances**2 / (2 * self.sigma**2))
        return activations

    def fit(self, X, y, lambda_reg=0.1):
        print("Ajustando la red RBF con regularización...")
        num_samples = min(self.num_centers, X.shape[0])  # Ajuste del número de centros
        indices = np.random.choice(X.shape[0], num_samples, replace=False)
        self.centers = X[indices]
        activations = self._calculate_activations(X)
        Q, R = np.linalg.qr(activations)
        R_reg = R + lambda_reg * np.eye(R.shape[0])  # Añadir término de regularización
        self.weights = np.linalg.solve(R_reg, Q.T @ y)
        print("Ajuste completado.")

    def predict(self, X):
        activations = self._calculate_activations(X)
        return activations @ self.weights

# Cargar datos
print("Cargando datos...")
file_path_1bit = "./10x10/10x10/1bit/DATASET_v7.mat"
file_path_2bit = "./10x10/10x10/2bit/DATASET_v7.mat"

mat_data1b = scipy.io.loadmat(file_path_1bit)
mat_data2b = scipy.io.loadmat(file_path_2bit)
print("Datos cargados.")

# Obtenemos los dataset
dataset1b = mat_data1b['DATASET']
dataset2b = mat_data2b['DATASET']

# Acceder a las variables dentro del archivo (1bit)
labels_1b = dataset1b['labels']
inputs_1b = dataset1b['input']
bits_1b = dataset1b['bits']
beams_1b = dataset1b['beams']

# Acceder a las variables dentro del archivo (2bit)
labels_2b = dataset2b['labels']
inputs_2b = dataset2b['input']
bits_2b = dataset2b['bits']
beams_2b = dataset2b['beams']

# Filtrar datos de 1 bit
print("Filtrando datos de 1 bit...")
datos1b_filtrados_input, datos1b_filtrados_labels = filtrar_datos_en_rango(inputs_1b, labels_1b)

# Filtrar datos de 2 bits
print("Filtrando datos de 2 bits...")
datos2b_filtrados_input, datos2b_filtrados_labels = filtrar_datos_en_rango(inputs_2b, labels_2b)

print("Datos filtrados listos.")

# Procesar los tableros
print("Procesando tableros...")

# 1 bit
tableros_procesados_1b = procesar_tableros(datos1b_filtrados_labels, 1)

# 2 bit
tableros_procesados_2b = procesar_tableros(datos2b_filtrados_labels, 2)

# Verificar dimensiones antes de transponer
print("Dimensiones antes de reshaping:")
print("tableros_procesados_1b.shape:", tableros_procesados_1b.shape)
print("tableros_procesados_2b.shape:", tableros_procesados_2b.shape)

# Asegurarse de que las dimensiones son correctas antes de transponer y reshaping
assert tableros_procesados_1b.ndim == 3, "tableros_procesados_1b debe ser un array de 3 dimensiones"
assert tableros_procesados_2b.ndim == 3, "tableros_procesados_2b debe ser un array de 3 dimensiones"

# Reshape tableros
rs_tableros_procesados_1b = tableros_procesados_1b.reshape(tableros_procesados_1b.shape[0], -1)
rs_tableros_procesados_2b = tableros_procesados_2b.reshape(tableros_procesados_2b.shape[0], -1)

print("Dimensiones después de reshaping:")
print("rs_tableros_procesados_1b.shape:", rs_tableros_procesados_1b.shape)
print("rs_tableros_procesados_2b.shape:", rs_tableros_procesados_2b.shape)

# Normalizar los datos
scaler = StandardScaler()
rs_tableros_procesados_1b = scaler.fit_transform(rs_tableros_procesados_1b)
rs_tableros_procesados_2b = scaler.fit_transform(rs_tableros_procesados_2b)

# Comprobar las dimensiones después del reshaping
print("Dimensiones de datos1b_filtrados_labels:", datos1b_filtrados_labels.shape)
print("Dimensiones de datos2b_filtrados_labels:", datos2b_filtrados_labels.shape)

# Ajustar las dimensiones de datos1b_filtrados_labels
reshaped_labels_1b = datos1b_filtrados_labels.reshape(-1, datos1b_filtrados_labels.shape[0] * datos1b_filtrados_labels.shape[1])
reshaped_labels_2b = datos2b_filtrados_labels.reshape(-1, datos2b_filtrados_labels.shape[0] * datos2b_filtrados_labels.shape[1])

# Comprobar las dimensiones después del reshaping
print("Dimensiones de reshaped_labels_1b:", reshaped_labels_1b.shape)
print("Dimensiones de reshaped_labels_2b:", reshaped_labels_2b.shape)

# Comprobar las dimensiones después del reshaping
print("Dimensiones de rs_tableros_procesados_1b:", rs_tableros_procesados_1b.shape)
print("Dimensiones de rs_tableros_procesados_2b:", rs_tableros_procesados_2b.shape)


# Asegurarse de que las dimensiones coinciden
assert rs_tableros_procesados_1b.shape[0] == reshaped_labels_1b.shape[0], "El número de muestras debe coincidir para 1 bit"
assert rs_tableros_procesados_2b.shape[0] == reshaped_labels_2b.shape[0], "El número de muestras debe coincidir para 2 bits"

# Inicializar matrices para almacenar métricas
predicciones_100_RBFs = np.zeros((10, 10, rs_tableros_procesados_1b.shape[0]))
mse_metrics = np.zeros((10, 10))
mae_metrics = np.zeros((10, 10))
r2_metrics = np.zeros((10, 10))

# Realizar validación cruzada y entrenar 100 RBFs
print("Realizando validación cruzada...")
for i in range(10):
    for j in range(10):
        print(f"Celda: ( {i}, {j} )")
        y_labels = reshaped_labels_1b[:, i * 10 + j]
        
        # Comprobar las dimensiones antes de la división
        print(f"Dimensiones de rs_tableros_procesados_1b: {rs_tableros_procesados_1b.shape}")
        print(f"Dimensiones de y_labels para celda ({i},{j}): {y_labels.shape}")

        # Crear estudio de Optuna y optimizar para cada celda
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        
        # Obtener los mejores hiperparámetros
        best_params = study.best_params
        print(f"Mejores hiperparámetros para celda ({i},{j}):", best_params)
        
        # Ajustar la RBF con los mejores hiperparámetros
        rbf_net = RBFNetwork(num_centers=best_params["num_centers"], sigma=best_params["sigma"])
        rbf_net.fit(rs_tableros_procesados_1b, y_labels, lambda_reg=best_params["lambda_reg"])
        predictions_all = rbf_net.predict(rs_tableros_procesados_1b)
        
        # Guardar predicciones y métricas
        predicciones_100_RBFs[i, j, :] = predictions_all.flatten()
        mse_metrics[i, j] = mean_squared_error(y_labels, predictions_all)
        mae_metrics[i, j] = mean_absolute_error(y_labels, predictions_all)
        r2_metrics[i, j] = r2_score(y_labels, predictions_all)

# Guardar resultados en un archivo Excel
import xlsxwriter

workbook = xlsxwriter.Workbook('resultados_rbf.xlsx')
worksheet = workbook.add_worksheet()

# Escribir encabezados
worksheet.write(0, 0, 'Celda')
worksheet.write(0, 1, 'MSE')
worksheet.write(0, 2, 'MAE')
worksheet.write(0, 3, 'R2')

# Escribir datos
for i in range(10):
    for j in range(10):
        worksheet.write(i * 10 + j + 1, 0, f'({i},{j})')
        worksheet.write(i * 10 + j + 1, 1, mse_metrics[i, j])
        worksheet.write(i * 10 + j + 1, 2, mae_metrics[i, j])
        worksheet.write(i * 10 + j + 1, 3, r2_metrics[i, j])

workbook.close()

# Graficar resultados en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(10), range(10))
ax.plot_surface(X, Y, mse_metrics, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('MSE')
plt.title('Superficie de MSE por celda')
plt.show()

# Graficar resultados en 2D
plt.figure(figsize=(10, 6))
for i in range(10):
    for j in range(10):
        plt.plot(mse_metrics[i, j], mae_metrics[i, j], marker='o', label=f'Celda ({i},{j})')
plt.xlabel('MSE')
plt.ylabel('MAE')
plt.title('MSE vs MAE por celda')
plt.legend()
plt.grid(True)
plt.show()
