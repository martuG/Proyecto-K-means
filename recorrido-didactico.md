Claro. Para planificar la clase, plantearía el proyecto como una **introducción práctica a Machine Learning no supervisado mediante K-Means**, utilizando un dataset de canciones para que el concepto sea intuitivo antes de entrar en la implementación.

## Descripción del proyecto

### Nombre del proyecto

**Agrupamiento de canciones mediante K-Means**

### Objetivo general

Introducir a los alumnos en el concepto de **aprendizaje no supervisado** mediante la implementación de un algoritmo K-Means en Python.

El proyecto busca que los alumnos comprendan, antes que memorizar código, **qué problema resuelve K-Means, cómo determina los grupos y cómo podemos interpretar los resultados obtenidos**.

### Situación problemática

Una plataforma de música dispone de información sobre distintas canciones, pero no tiene una clasificación previa que indique a qué tipo de canción pertenece cada una.

Cada canción posee determinadas características numéricas, por ejemplo:

- Energía.
- Bailabilidad.
- Acústica.
- Tempo.

La plataforma quiere encontrar automáticamente grupos de canciones que tengan características similares.

La pregunta que guía el proyecto es:

> **¿Podemos hacer que una computadora descubra grupos de canciones similares sin indicarle previamente cuáles son esos grupos?**

Esto permite introducir naturalmente el concepto de **Machine Learning no supervisado**.

---



## Enfoque didáctico

Como los alumnos todavía no tienen conocimientos previos de Machine Learning, el notebook debería avanzar progresivamente.

Primero se plantea el problema sin hablar de algoritmos.

Luego se presenta un pequeño conjunto de canciones y se les pide que observen los datos y piensen:

> Si tuvieran que separar estas canciones en tres grupos, ¿cuáles pondrían juntas?

A partir de esa actividad se introduce la idea de **agrupamiento o clustering**.

Después se explica K-Means conceptualmente:

1. Elegir la cantidad de grupos `K`.
2. Seleccionar centroides iniciales.
3. Calcular la distancia de cada punto a los centroides.
4. Asignar cada punto al centroide más cercano.
5. Recalcular los centroides.
6. Repetir el proceso hasta alcanzar una situación estable.

Finalmente se implementa el algoritmo utilizando Python y `scikit-learn`.

---



# Estructura propuesta del notebook



## 1. Introducción al problema



### Markdown

Presentación de la situación:

> Tenemos un conjunto de canciones y algunas características que describen cómo son.
>
> No conocemos previamente las categorías a las que pertenecen.
>
> Nuestro objetivo será encontrar automáticamente grupos de canciones similares.

Pregunta para los alumnos:

> ¿Cómo agruparían las canciones si tuvieran que hacerlo manualmente?

---

# 2. ¿Qué es Machine Learning?

Una introducción muy breve.

### Conceptos a presentar

- Machine Learning.
- Datos.
- Características.
- Modelo.
- Aprendizaje supervisado.
- Aprendizaje no supervisado.

No profundizar todavía en algoritmos.

La idea central:

> En el aprendizaje supervisado conocemos previamente la respuesta que queremos predecir. En el aprendizaje no supervisado buscamos descubrir estructuras o patrones dentro de los datos.

Conectar directamente con el problema:

> En nuestro caso no sabemos previamente qué grupo corresponde a cada canción.

---



# 3. Nuestro dataset

Crear un dataset pequeño de canciones.

Inicialmente utilizar solamente:

- `energia`
- `bailabilidad`

Ejemplo:


| canción   | energía | bailabilidad |
| --------- | ------- | ------------ |
| Canción A | 0.90    | 0.88         |
| Canción B | 0.85    | 0.92         |
| Canción C | 0.25    | 0.30         |
| Canción D | 0.32    | 0.25         |
| Canción E | 0.65    | 0.45         |




### Código

```python
import pandas as pd

datos = {
    "cancion": [
        "Canción A",
        "Canción B",
        "Canción C",
        "Canción D",
        "Canción E"
    ],
    "energia": [
        0.90,
        0.85,
        0.25,
        0.32,
        0.65
    ],
    "bailabilidad": [
        0.88,
        0.92,
        0.30,
        0.25,
        0.45
    ]
}

df = pd.DataFrame(datos)

df
```

---



# 4. Exploración de los datos

Mostrar:

```python
df.head()
```

```python
df.describe()
```

Y explicar:

- filas;
- columnas;
- características;
- valores numéricos.

Introducir el concepto de **feature** como característica utilizada por el algoritmo.

---



# 5. Visualización de las canciones

Esta parte es fundamental para la clase.

Representar:

- eje X: energía
- eje Y: bailabilidad

```python
import matplotlib.pyplot as plt

plt.scatter(df["energia"], df["bailabilidad"])

plt.xlabel("Energía")
plt.ylabel("Bailabilidad")
plt.title("Canciones según energía y bailabilidad")

plt.show()
```

Después plantear:

> Mirando el gráfico, ¿ven grupos?

Y:

> Si tuvieran que formar tres grupos, ¿dónde los ubicarían?

---



# 6. Introducción al clustering

Explicar que el objetivo es encontrar grupos cuyos elementos sean similares entre sí.

Conceptos:

### Clustering

Agrupar elementos similares.

### Cluster

Cada uno de los grupos encontrados.

### Centroide

El punto que representa el centro de un cluster.

### K

Cantidad de clusters que queremos obtener.

---



# 7. ¿Qué significa K-Means?

Explicar el nombre:

**K** → cantidad de grupos.

**Means** → medias/promedios utilizados para calcular los centroides.

Plantear:

> Si queremos obtener tres grupos, ¿qué valor tendrá K?

```python
K = 3
```

---



# 8. Funcionamiento conceptual del algoritmo

Antes de usar `sklearn`, mostrar el algoritmo de manera visual.

### Paso 1: elegir K

```text
K = 3
```



### Paso 2: seleccionar centroides iniciales

Mostrar los puntos y tres centroides.

### Paso 3: asignar cada canción al centroide más cercano

Explicar la idea de distancia.

### Paso 4: recalcular los centroides

El nuevo centroide se obtiene a partir de los puntos pertenecientes al grupo.

### Paso 5: repetir

Hasta que los grupos se estabilicen.

Esta parte puede hacerse con gráficos sucesivos para que los alumnos vean el movimiento de los centroides.

---



# 9. Distancia entre puntos

Introducir brevemente la **distancia euclídea**.

Para dos puntos:

```text
A = (x1, y1)
B = (x2, y2)
```

La distancia:

```text
d = √((x2 - x1)² + (y2 - y1)²)
```

No hace falta profundizar matemáticamente si el objetivo de la clase es una introducción.

La idea importante:

> K-Means utiliza la distancia para determinar qué centroide está más cerca de cada dato.

---



# 10. Implementación con Scikit-Learn

Instalar/importar:

```python
from sklearn.cluster import KMeans
```

Preparar las variables:

```python
X = df[["energia", "bailabilidad"]]
```

Crear el modelo:

```python
modelo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

Entrenar:

```python
modelo.fit(X)
```

---



# 11. Obtener los grupos

```python
df["cluster"] = modelo.labels_

df
```

Explicar:

> El algoritmo acaba de asignar cada canción a uno de los tres clusters.

Importante:

Los números `0`, `1` y `2` **no representan categorías reales**.

El cluster `0` no significa "mejor" que el cluster `1`.

Son simplemente identificadores.

---



# 12. Visualizar los resultados

```python
plt.scatter(
    df["energia"],
    df["bailabilidad"],
    c=df["cluster"]
)

plt.xlabel("Energía")
plt.ylabel("Bailabilidad")
plt.title("Clusters de canciones")

plt.show()
```

Luego mostrar los centroides:

```python
centroides = modelo.cluster_centers_

plt.scatter(
    df["energia"],
    df["bailabilidad"],
    c=df["cluster"]
)

plt.scatter(
    centroides[:, 0],
    centroides[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Energía")
plt.ylabel("Bailabilidad")
plt.title("Clusters y centroides")

plt.show()
```

---



# 13. Interpretación de los clusters

Esta sección es importante porque permite pasar de la programación a la interpretación.

```python
df.groupby("cluster")[["energia", "bailabilidad"]].mean()
```

Preguntas:

- ¿Qué características tiene cada grupo?
- ¿Qué cluster contiene canciones más bailables?
- ¿Qué cluster tiene mayor energía?
- ¿Cómo podríamos describir cada grupo?

La idea es que los alumnos comprendan que **el algoritmo encuentra los grupos, pero nosotros interpretamos qué significan**.

---



# 14. ¿Qué pasa si elegimos otro K?

Ejecutar nuevamente con:

```python
K = 2
```

Luego:

```python
K = 4
```

Comparar los resultados.

Preguntar:

> ¿Cuál parece ser el mejor número de clusters?

Esto permite introducir uno de los principales problemas de K-Means:

> **¿Cómo sabemos qué valor de K elegir?**

---



# 15. Método del codo

Introducir el concepto de **inercia**.

Ejecutar K-Means para diferentes valores de K:

```python
inercia = []

for k in range(1, 10):
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    modelo.fit(X)
    
    inercia.append(modelo.inertia_)
```

Graficar:

```python
plt.plot(range(1, 10), inercia, marker="o")

plt.xlabel("Cantidad de clusters (K)")
plt.ylabel("Inercia")
plt.title("Método del codo")

plt.show()
```

Explicar que buscamos un punto donde agregar más clusters deja de producir una mejora significativa.

---



# 16. Actividad para los alumnos

Esta parte puede quedar preparada para que ellos modifiquen el notebook.

### Actividad 1

Cambiar:

```python
n_clusters=3
```

por:

```python
n_clusters=2
```

y analizar el resultado.

### Actividad 2

Probar:

```python
n_clusters=4
```



### Actividad 3

Comparar los resultados.

Responder:

1. ¿Qué ocurre cuando aumenta K?
2. ¿Qué pasa con el tamaño de los clusters?
3. ¿Qué valor de K elegirían?
4. ¿Por qué?

---



# 17. Extensión: agregar más características

Una vez entendido el ejemplo de dos dimensiones, ampliar el problema.

Agregar:

- `acustica`
- `tempo`

Por ejemplo:

```python
X = df[
    [
        "energia",
        "bailabilidad",
        "acustica",
        "tempo"
    ]
]
```

Explicar:

> Antes utilizábamos dos características porque podíamos representarlas fácilmente en un gráfico. K-Means no está limitado a dos dimensiones.

Esto es una excelente transición hacia problemas reales.

---



# 18. Normalización de datos

Acá introduciría `StandardScaler`, pero **solo como extensión**.

El problema:

- energía puede estar entre 0 y 1;
- bailabilidad entre 0 y 1;
- tempo puede estar entre 60 y 180.

Por lo tanto, una variable puede tener una escala mucho mayor que otra.

Introducir:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Y posteriormente:

```python
modelo.fit(X_scaled)
```

La idea que deberían llevarse:

> Cuando las características tienen escalas muy diferentes, debemos prestar atención a la escala de los datos porque K-Means utiliza distancias.

---



# 19. ¿Qué aprendimos?

Cerrar el notebook recuperando los conceptos principales:

```text
Datos
  ↓
Características
  ↓
Distancias
  ↓
K-Means
  ↓
Clusters
  ↓
Centroides
  ↓
Interpretación
```



### Conceptos

- Machine Learning.
- Aprendizaje no supervisado.
- Clustering.
- Cluster.
- K.
- Centroide.
- Distancia.
- K-Means.
- Inercia.
- Método del codo.
- Normalización.

---



# 20. Preguntas de cierre

Podés terminar la clase con preguntas como:

1. ¿Por qué K-Means es un algoritmo de aprendizaje no supervisado?
2. ¿Qué representa K?
3. ¿Qué es un centroide?
4. ¿Cómo decide K-Means a qué cluster pertenece un dato?
5. ¿Qué ocurre cuando modificamos K?
6. ¿Por qué los clusters `0`, `1` y `2` no tienen un significado específico?
7. ¿Para qué sirve el método del codo?
8. ¿Por qué puede ser necesario normalizar los datos?
9. ¿Qué pasaría si agregáramos muchas características?
10. ¿Qué diferencia habría entre decirle al algoritmo el género de cada canción y dejar que encuentre los grupos por sí mismo?



## Secuencia pedagógica recomendada

Para una clase introductoria, yo la organizaría así:

**Problema real → datos → visualización → intuición de agrupamiento → concepto de K-Means → algoritmo paso a paso → implementación → resultados → interpretación → elección de K → extensión.**

Y mantendría **la implementación de K-Means manual como explicación conceptual**, pero usaría `scikit-learn` para la implementación real. Así los alumnos no quedan con la idea de que Machine Learning consiste simplemente en llamar a `KMeans().fit()` sin comprender qué está sucediendo internamente.