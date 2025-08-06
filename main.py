from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from io import BytesIO
import base64
import os
from datetime import datetime
from collections import Counter

# Initialisation de l'app
app = FastAPI(title="API Analyse Veille Médiatique")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Palette et ordre des sentiments
palette_custom = ["#81C3D7", "#219ebc", "#D9DCD6", "#2F6690", "#16425B"]
desired_order = ['strongly positive', 'positive', 'neutral', 'negative', 'strongly negative']

# Fonction d'encodage des graphiques
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Pydantic model pour la requête JSON
class Article(BaseModel):
    author: str
    content_excerpt: str
    published_at: int
    sentiment_label: str
    title: str
    source_link: str
    keywords: list[str] = []

class JSONData(BaseModel):
    data: list[Article]

# Route POST /analyser_json
@app.post("/analyser_json")
async def analyser_json(payload: JSONData):
    raw_data = payload.data
    df = pd.DataFrame([article.dict() for article in raw_data])

    df["articleCreatedDate"] = df["published_at"].apply(lambda ts: datetime.utcfromtimestamp(ts))
    df = df.rename(columns={
        "author": "authorName",
        "sentiment_label": "sentimentHumanReadable",
    })

    df['Year'] = df['articleCreatedDate'].dt.year

    kpis = {
        "total_mentions": int(df.shape[0]),
        "positive": int(df[df['sentimentHumanReadable'] == 'positive'].shape[0]),
        "negative": int(df[df['sentimentHumanReadable'] == 'negative'].shape[0]),
        "neutral": int(df[df['sentimentHumanReadable'] == 'neutral'].shape[0]),
    }

    df['Period'] = df['articleCreatedDate'].dt.date

    # Évolution des mentions
    mentions_over_time = df['Period'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(mentions_over_time.index.astype(str), mentions_over_time.values, marker='o', linestyle='-', color="#2F6690")
    ax1.set_title("Évolution des mentions par jour")
    ax1.set_ylabel("Mentions")
    plt.xticks(rotation=45)
    evolution_mentions_b64 = fig_to_base64(fig1)
    plt.close(fig1)

    # WordCloud des mots-clés
    all_keywords = [kw for sublist in df["keywords"] if isinstance(sublist, list) for kw in sublist]
    if all_keywords:
        keywords_text = " ".join(all_keywords)
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap='Blues').generate(keywords_text)
        fig_kw, ax_kw = plt.subplots(figsize=(10, 5))
        ax_kw.imshow(wordcloud, interpolation='bilinear')
        ax_kw.axis("off")
        ax_kw.set_title("Mots-clés les plus fréquents", fontsize=16)
        keywords_b64 = fig_to_base64(fig_kw)
        plt.close(fig_kw)
    else:
        keywords_b64 = ""

    # Répartition sentiments par auteur
    author_sentiment = df.groupby(['authorName', 'sentimentHumanReadable']).size().unstack(fill_value=0)
    author_sentiment['Total'] = author_sentiment.sum(axis=1)
    top_authors_sentiment = author_sentiment.sort_values(by='Total', ascending=False).head(10).drop(columns='Total')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    top_authors_sentiment.plot(kind='barh', stacked=True, ax=ax3, color="#2F6690")
    ax3.set_xlabel("Nombre d'articles")
    ax3.set_ylabel("Auteur")
    ax3.set_title("Répartition des sentiments par auteur")
    sentiments_auteurs_b64 = fig_to_base64(fig3)
    plt.close(fig3)

    # Tableau top auteurs
    top_table = (
        df['authorName']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'count', 'authorName': 'Auteur'})
        .head(10)
        .to_html(index=False, border=1, classes="styled-table")
    )

    # Rapport HTML
    html_report = f"""<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='UTF-8'>
    <title>Rapport de Veille Médiatique</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 40px; max-width: 900px; margin: auto; background-color: #f9f9f9; }}
        h1, h2 {{ text-align: center; color: #2F6690; }}
        .centered-text {{ max-width: 800px; margin: 0 auto 40px; text-align: center; font-size: 16px; line-height: 1.6; }}
        .styled-table {{ border-collapse: collapse; margin: 25px auto; font-size: 16px; width: 80%; border: 1px solid #dddddd; }}
        .styled-table th, .styled-table td {{ padding: 10px 15px; text-align: left; border: 1px solid #dddddd; }}
        .styled-table thead th {{ background-color: #f3f3f3; font-weight: bold; }}
        .image-block {{ text-align: center; margin: 30px 0; }}
    </style>
</head>
<body>
    <h1>📊 Rapport d'Analyse de Veille Médiatique</h1>
    <div class="centered-text">
        <p>Ce rapport présente une analyse des articles provenant de Lumenfeed, avec des statistiques sur la couverture médiatique, les sentiments exprimés, et les auteurs les plus actifs.</p>
    </div>
    <h2>Indicateurs Clés</h2>
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="text-align: center;"><h3>{kpis['total_mentions']}</h3><p>Mentions totales</p></div>
        <div style="text-align: center;"><h3>{kpis['positive']}</h3><p>Positives</p></div>
        <div style="text-align: center;"><h3>{kpis['negative']}</h3><p>Négatives</p></div>
        <div style="text-align: center;"><h3>{kpis['neutral']}</h3><p>Neutres</p></div>
    </div>
    <div class="image-block">
        <h2>Évolution des mentions</h2>
        <img src="data:image/png;base64,{evolution_mentions_b64}" width="700"/>
    </div>
    <div class="image-block">
        <h2>Mots-clés les plus fréquents</h2>
        <img src="data:image/png;base64,{keywords_freq_b64}" width="600"/>
    </div>
    <div class="image-block">
        <h2>Répartition des sentiments par auteur</h2>
        <img src="data:image/png;base64,{sentiments_auteurs_b64}" width="700"/>
    </div>
    <h2>Top 10 Auteurs les plus actifs</h2>
    {top_table}
</body>
</html>
"""

    os.makedirs("static", exist_ok=True)
    with open("static/rapport_veille.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    return {
        "kpis": kpis,
        "html_report": html_report
    }

# Route GET pour consulter le rapport
@app.get("/rapport")
def get_rapport():
    return FileResponse("static/rapport_veille.html", media_type="text/html")
