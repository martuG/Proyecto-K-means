# Agrupamiento de canciones mediante K-Means

Proyecto introductorio a **Machine Learning no supervisado** con K-Means.

## Dataset

Se utiliza un subconjunto didáctico del dataset público:

- **Nombre:** Spotify Tracks Dataset
- **Autor:** Maharshi Pandya
- **Fuente:** [https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)

El archivo listo para la clase está en:

- `datos/canciones_spotify.csv`

Columnas:


| Columna        | Descripción                                                            |
| -------------- | ---------------------------------------------------------------------- |
| `cancion`      | Nombre de la pista                                                     |
| `artista`      | Artista(s)                                                             |
| `energia`      | Intensidad percibida (0–1)                                             |
| `bailabilidad` | Qué tan bailable es (0–1)                                              |
| `acustica`     | Qué tan acústica es (0–1)                                              |
| `tempo`        | Tempo estimado en BPM                                                  |
| `genero`       | Género original (solo para interpretar; no se usa en el entrenamiento) |


## Cómo ejecutar

```bash
pip install -r requirements.txt
python -m jupyter notebook kmeans_canciones.ipynb
```

- `recorrido-didactico.md` — plan de clase
- `kmeans_canciones.ipynb` — notebook completo de la clase
- `datos/canciones_spotify.csv` — dataset limpio para usar en clase
- `requirements.txt` — dependencias

## Secuencia pedagógica

Problema real → datos → visualización → intuición de agrupamiento → concepto de K-Means → implementación con scikit-learn → interpretación → método del codo → extensión con dataset real y normalización.