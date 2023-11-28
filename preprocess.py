from params import USEFUL_FEATS

"""
Garde seulement les espèces françaises et les colonnes intéressantes. Cela fait
2706 lignes et 7 colonnes. Il reste 257 na dans le playback_used, et 2 dans le bitrate
"""
def clean_data(df):
    species_fr = df[df["country"]== "France"]["species"].unique()
    df_clean = (df[df["species"].isin(species_fr)])[USEFUL_FEATS]
    return df_clean
