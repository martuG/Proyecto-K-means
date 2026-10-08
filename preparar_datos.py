"""
Descarga el dataset público de Spotify y genera el CSV didáctico
usado en la clase: datos/canciones_spotify.csv
"""

from pathlib import Path

import pandas as pd
import urllib.request

URL = (
    "https://huggingface.co/datasets/maharshipandya/"
    "spotify-tracks-dataset/resolve/main/dataset.csv"
)

GENEROS = [
    "acoustic",
    "classical",
    "jazz",
    "ambient",
    "dance",
    "disco",
    "edm",
    "techno",
    "rock",
    "metal",
    "hard-rock",
    "hip-hop",
    "reggae",
    "latin",
]


def main():
    raw_path = Path("dataset_spotify_raw.csv")
    out_dir = Path("datos")
    out_path = out_dir / "canciones_spotify.csv"

    if not raw_path.exists():
        print("Descargando dataset público...")
        urllib.request.urlretrieve(URL, raw_path)
        print("Descarga completa.")

    df = pd.read_csv(raw_path)
    sub = df[df["track_genre"].isin(GENEROS)].copy()
    sub = sub.dropna(
        subset=["track_name", "artists", "energy", "danceability", "acousticness", "tempo"]
    )
    sub = sub.drop_duplicates(subset=["track_id"])

    muestra = (
        sub.groupby("track_genre", group_keys=False)
        .sample(n=40, random_state=42)
        .reset_index(drop=True)
    )

    out = pd.DataFrame(
        {
            "cancion": muestra["track_name"],
            "artista": muestra["artists"],
            "energia": muestra["energy"].round(3),
            "bailabilidad": muestra["danceability"].round(3),
            "acustica": muestra["acousticness"].round(3),
            "tempo": muestra["tempo"].round(1),
            "genero": muestra["track_genre"],
        }
    )

    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Dataset didáctico guardado en {out_path} ({len(out)} canciones)")


if __name__ == "__main__":
    main()
