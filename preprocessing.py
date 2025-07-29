import pandas as pd

def clean_dataframe(df):
    # 🔹 Dates
    df['articleCreatedDate'] = pd.to_datetime(df['articleCreatedDate'], errors='coerce')

    # 🔹 Colonnes inutiles
    cols_to_drop = [
        'articleJSON', 'mainCategoryID', 'processingCost', 'publisher',
        'totalTokens', 'outputTokens', 'inputTokens',
        'videoType', 'videoURL',
        'typesenseID', 'typesenseCollection', '__v'
    ] + [f'keywords[{i}]' for i in range(24) if f'keywords[{i}]' in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 🔹 Lignes inexploitables
    df.dropna(subset=['articleCreatedDate'], inplace=True)
    df.dropna(subset=['articleTitle', 'articleDescription', 'articleCleanDescription'], inplace=True)

    # 🔹 Auteurs
    df['authorName'] = df['authorName'].fillna("Inconnu").astype(str).str.strip()

    # 🔹 Nettoyage des sentiments
    df['sentimentHumanReadable'] = df['sentimentHumanReadable'].astype(str).str.strip().str.lower()
    df = df[df['sentimentHumanReadable'].isin([
        'positive', 'negative', 'neutral', 'strongly positive', 'strongly negative'
    ])]

    # 🔹 Réinitialisation
    df.reset_index(drop=True, inplace=True)

    return df
