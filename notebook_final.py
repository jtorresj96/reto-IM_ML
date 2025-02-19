#%%
# ============================================================================
# 1. Importación de librerías
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

#%%
# ============================================================================
# 2. Carga de datos
# ============================================================================

train_df = pd.read_csv("archive/train.csv")
test_df  = pd.read_csv("archive/test.csv")

# Mostrar las primeras filas de train
print("Primeras filas del dataset de entrenamiento:")
print(train_df.head())
#%%
# ============================================================================
# 3. Análisis Exploratorio de Datos (EDA) sobre train_df
# ============================================================================
print("\nInformación general del dataset de entrenamiento:")
print(train_df.info())

print("\nEstadísticas descriptivas de Weekly_Sales:")
print(train_df['Weekly_Sales'].describe())

# Convertir la columna 'Date' a datetime y ordenar
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df = train_df.sort_values('Date')

# Graficar la serie de ventas a lo largo del tiempo
plt.figure(figsize=(12,6))
plt.plot(train_df['Date'], train_df['Weekly_Sales'], label="Ventas Semanales")
plt.title('Ventas Semanales a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('Ventas Semanales')
plt.legend()
plt.show()

# Histograma con KDE para Weekly_Sales
plt.figure(figsize=(8,4))
sns.histplot(train_df['Weekly_Sales'], kde=True)
plt.title('Distribución de Ventas Semanales')
plt.xlabel('Ventas')
plt.show()

# Boxplot para detectar outliers en Weekly_Sales
plt.figure(figsize=(8,4))
sns.boxplot(x=train_df['Weekly_Sales'])
plt.title('Boxplot de Ventas Semanales')
plt.xlabel('Ventas')
plt.show()
#%%
# ============================================================================
# 4. Preprocesamiento y Feature Engineering
# ============================================================================

# 4.1. Preparar ambos datasets para aplicar las mismas transformaciones
# Agregamos una columna para distinguir el origen de los datos
train_df['dataset'] = 'train'
test_df['dataset']  = 'test'

# En test_df, la columna Weekly_Sales no existe (ya que se debe predecir)
# Para facilitar el feature engineering, creamos la columna con NaN
test_df['Weekly_Sales'] = np.nan

# Convertir la columna 'Date' a datetime en test_df y ordenar
test_df['Date'] = pd.to_datetime(test_df['Date'])
# Unir ambos datasets para aplicar transformaciones de manera uniforme
all_data = pd.concat([train_df, test_df], sort=False)
all_data = all_data.sort_values('Date')

# 4.2. Variables derivadas de la fecha
all_data['Year']  = all_data['Date'].dt.year
all_data['Month'] = all_data['Date'].dt.month

# 4.3. Variables de retardo (lag features) y medias móviles
# Estas se calcularán por grupo (Store y Dept) ordenando por fecha.
# Para la columna Weekly_Sales: en los datos de test, al no haber valor,
# se obtendrá el valor de la última observación del grupo (de train), si existe.
all_data['lag_1'] = all_data.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)

# Media móvil de las últimas 4 semanas por grupo.
all_data['rolling_mean_4'] = all_data.groupby(['Store', 'Dept'])['Weekly_Sales']\
                                       .transform(lambda x: x.rolling(window=4, min_periods=1).mean())

# 4.4. Codificación de variables categóricas
# Convertir IsHoliday a entero (0/1)
all_data['IsHoliday'] = all_data['IsHoliday'].astype(int)

# Aplicar One-Hot Encoding para las columnas 'Store' y 'Dept'
all_data = pd.get_dummies(all_data, columns=['Store', 'Dept'], drop_first=True)
#%%
# ============================================================================
# 5. Separar nuevamente en dataset de entrenamiento y dataset final de test
# ============================================================================
# train_data: datos con target (Weekly_Sales no nulo)
train_data = all_data[all_data['dataset'] == 'train'].copy()
# test_final: datos para predecir (Weekly_Sales es NaN)
test_final = all_data[all_data['dataset'] == 'test'].copy()
#%%
# ============================================================================
# 6. División del dataset de entrenamiento en train y validación (time-based)
# ============================================================================
# Definir una fecha de corte para separar entrenamiento y validación
# (por ejemplo, usar datos anteriores a 2012 para entrenar y el resto para validar)
validation_cutoff = '2012-01-01'
train_train = train_data[train_data['Date'] < validation_cutoff].copy()
train_val   = train_data[train_data['Date'] >= validation_cutoff].copy()

print("\nTamaño del dataset de entrenamiento (train_train):", train_train.shape)
print("Tamaño del dataset de validación (train_val):", train_val.shape)
#%%
# ============================================================================
# 7. Selección de Features y definición del target
# ============================================================================
# Lista de features a usar (ajusta según las variables que consideres importantes)
features = ['Year', 'Month', 'lag_1', 'rolling_mean_4', 'IsHoliday']
# Incluir las columnas generadas por One-Hot Encoding (por ejemplo, que empiezan con 'Store_' o 'Dept_')
features += [col for col in all_data.columns if col.startswith('Store_') or col.startswith('Dept_')]

# Variables predictoras y target para entrenamiento y validación
X_train = train_train[features]
y_train = train_train['Weekly_Sales']

X_val   = train_val[features]
y_val   = train_val['Weekly_Sales']
#%%
# ============================================================================
# 8. Entrenamiento y Evaluación del Modelo
# ============================================================================
# Inicializar y entrenar un modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones en el conjunto de validación
val_preds = model.predict(X_val)

# Evaluación del modelo en validación
mae = mean_absolute_error(y_val, val_preds)

print("\nResultados en el conjunto de validación:")
print("MAE:", mae)

#%%
# ============================================================================
# 9. Predicción en el Dataset Final de Test
# ============================================================================
# En test_final no se tiene Weekly_Sales (target); por lo tanto, se predicen esas ventas.
# Asegurarse de que el dataset final tenga las mismas features (ya creadas)
X_test_final = test_final[features]
final_predictions = model.predict(X_test_final)

# Añadir las predicciones al dataset final
test_final['Weekly_Sales_Predicted'] = final_predictions

# Mostrar algunas predicciones
print("\nEjemplo de predicciones en el dataset final:")
print(test_final[['Date', 'Weekly_Sales_Predicted']].head())
# %%
