from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import datetime
from typing import List

app = FastAPI(title="API Analyse Veille Médiatique")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

desired_order = ['strongly positive', 'positive', 'neutral', 'negative', 'strongly negative']
palette_custom = ["#81C3D7", "#219ebc", "#D9DCD6", "#2F6690", "#16425B"]

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def clean_json_data(data: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)

    # Renommer les colonnes pour correspondre à l'ancien schéma
    df.rename(columns={
        'author': 'authorName',
        'sentiment_label': 'sentimentHumanReadable',
        'published_at': 'articleCreatedDate'
    }, inplace=True)

    # Convertir la date en datetime
    df['articleCreatedDate'] = pd.to_datetime(df['articleCreatedDate'], unit='s', errors='coerce')

    # Supprimer les lignes avec des dates invalides
    df.dropna(subset=['articleCreatedDate'], inplace=True)

    return df

@app.post("/analyser_json")
async def analyser_json(request: Request, granularity: str = Form("Par mois")):
    body = await request.json()
    data = body if isinstance(body, list) else body.get("articles", [])

    df = clean_json_data(data)
    df['Year'] = df['articleCreatedDate'].dt.year

    kpis = {
        "total_mentions": int(df.shape[0]),
        "positive": int(df[df['sentimentHumanReadable'] == 'positive'].shape[0]),
        "negative": int(df[df['sentimentHumanReadable'] == 'negative'].shape[0]),
        "neutral": int(df[df['sentimentHumanReadable'] == 'neutral'].shape[0]),
    }

    if granularity == "Par jour":
        df['Period'] = df['articleCreatedDate'].dt.date
    elif granularity == "Par semaine":
        df['Period'] = df['articleCreatedDate'].dt.to_period('W')
    elif granularity == "Par mois":
        df['Period'] = df['articleCreatedDate'].dt.to_period('M')
    elif granularity == "Par année":
        df['Period'] = df['articleCreatedDate'].dt.to_period('Y')
    else:
        df['Period'] = df['articleCreatedDate'].dt.to_period('M')

    mentions_over_time = df['Period'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(mentions_over_time.index.astype(str), mentions_over_time.values, marker='o', linestyle='-', color="#2F6690")
    ax1.set_title(f"Évolution des mentions ({granularity.lower()})")
    ax1.set_xlabel("Période")
    ax1.set_ylabel("Mentions")
    plt.xticks(rotation=45)
    evolution_mentions_b64 = fig_to_base64(fig1)
    plt.close(fig1)

    sentiment_counts_raw = df['sentimentHumanReadable'].value_counts()
    sentiment_counts = pd.Series([sentiment_counts_raw.get(s, 0) for s in desired_order], index=desired_order)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=palette_custom, ax=ax2)
    ax2.set_ylabel("Nombre d'articles")
    ax2.set_xlabel("Sentiment")
    ax2.set_title("Répartition globale des sentiments")
    sentiments_global_b64 = fig_to_base64(fig2)
    plt.close(fig2)

    author_sentiment = df.groupby(['authorName', 'sentimentHumanReadable']).size().unstack(fill_value=0)
    author_sentiment['Total'] = author_sentiment.sum(axis=1)
    top_authors_sentiment = author_sentiment.sort_values(by='Total', ascending=False).head(10).drop(columns='Total')
    existing_sentiments = [s for s in desired_order if s in top_authors_sentiment.columns]
    top_authors_sentiment = top_authors_sentiment[existing_sentiments].iloc[::-1]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    top_authors_sentiment.plot(kind='barh', stacked=True, ax=ax3,
        color=[palette_custom[desired_order.index(s)] for s in existing_sentiments])
    ax3.set_xlabel("Nombre d'articles")
    ax3.set_ylabel("Auteur / Source")
    ax3.set_title("Répartition des sentiments par auteur")
    sentiments_auteurs_b64 = fig_to_base64(fig3)
    plt.close(fig3)

    top_table = (
        df['authorName']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'countt', 'authorName': 'Auteur'})
        .head(10)
        .to_html(index=False, border=1, classes="styled-table")
    )

    html_report = f"""<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='UTF-8'>
    <title>Rapport de Veille Médiatique</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 40px;
            max-width: 900px;
            margin: auto;
            background-color: #f9f9f9;
        }}
        h1, h2 {{
            text-align: center;
            color: #2F6690;
        }}
        .centered-text {{
            max-width: 800px;
            margin: 0 auto 40px;
            text-align: center;
            font-size: 16px;
            line-height: 1.6;
        }}
        .styled-table {{
            border-collapse: collapse;
            margin: 25px auto;
            font-size: 16px;
            width: 80%;
            border: 1px solid #dddddd;
        }}
        .styled-table th, .styled-table td {{
            padding: 10px 15px;
            text-align: left;
            border: 1px solid #dddddd;
        }}
        .styled-table thead th {{
            background-color: #f3f3f3;
            font-weight: bold;
        }}
        .image-block {{
            text-align: center;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <h1>📊 Rapport d'Analyse de Veille Médiatique</h1>

    <div class="centered-text">
        <p>
            Ce rapport de veille médiatique présente une analyse approfondie des articles publiés autour d’un sujet d’actualité.
            Il a pour objectif de fournir aux décideurs une vision claire et synthétique des mentions médiatiques, des perceptions exprimées
            (positives, négatives ou neutres) ainsi que des sources les plus influentes.
        </p>
    </div>

    <h2>Indicateurs Clés</h2>
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="text-align: center;">
            <h3>{kpis['total_mentions']}</h3>
            <p>Mentions totales</p>
        </div>
        <div style="text-align: center;">
            <h3>{kpis['positive']}</h3>
            <p>Positives</p>
        </div>
        <div style="text-align: center;">
            <h3>{kpis['negative']}</h3>
            <p>Négatives</p>
        </div>
        <div style="text-align: center;">
            <h3>{kpis['neutral']}</h3>
            <p>Neutres</p>
        </div>
    </div>

    <div class="image-block">
        <h2>Évolution des mentions</h2>
        <img src="data:image/png;base64,{evolution_mentions_b64}" width="700"/>
    </div>

    <div class="image-block">
        <h2>Répartition globale des sentiments</h2>
        <img src="data:image/png;base64,{sentiments_global_b64}" width="600"/>
    </div>

    <div class="image-block">
        <h2>Répartition des sentiments par auteur</h2>
        <img src="data:image/png;base64,{sentiments_auteurs_b64}" width="700"/>
    </div>

    <h2>Top 10 Auteurs les plus actifs</h2>
    {top_table}

</body>
</html>"""

    os.makedirs("static", exist_ok=True)
    with open("static/rapport_veille.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    return {
        "kpis": kpis,
        "html_report": html_report
    }

@app.get("/rapport")
def get_rapport():
    return FileResponse("static/rapport_veille.html", media_type="text/html")
