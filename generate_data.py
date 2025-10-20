import pandas as pd
from sklearn.datasets import load_iris

# Cargar dataset Iris
iris = load_iris()

# Crear DataFrame con las características
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Agregar la columna con la especie (para referencia)
df["species"] = [iris.target_names[i] for i in iris.target]

# Guardar el dataset completo
df.to_csv("iris_full.csv", index=False)

# También guardamos solo los datos numéricos (para usar en C)
df_numeric = df.drop(columns=["species"])
df_numeric.to_csv("iris_numeric.csv", index=False)

print("Archivos generados:")
print("- iris_full.csv  (con especies)")
print("- iris_numeric.csv  (solo datos numéricos)")
