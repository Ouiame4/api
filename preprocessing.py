# preprocessing.py

import pandas as pd

def clean_dataframe(df):
    df['articleCreatedDate'] = pd.to_datetime(df['articleCreatedDate'], errors='coerce')

    cols_to_drop = [
        'articleJSON', 'mainCategoryID', 'processingCost', 'publisher',
        'totalTokens', 'outputTokens', 'inputTokens',
        'videoType', 'videoURL',
        'typesenseID', 'typesenseCollection', '__v',
    ] + [f'keywords[{i}]' for i in range(24) if f'keywords[{i}]' in df.columns]

    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.dropna(subset=['articleCreatedDate'], inplace=True)
    df.dropna(subset=['articleTitle', 'articleDescription', 'articleCleanDescription'], inplace=True)
    df['authorName'].fillna("Inconnu", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
