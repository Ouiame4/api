
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os

app = FastAPI(title="API Analyse Veille Médiatique (Render + Rapport)")

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

@app.post("/analyser")
async def analyser_csv(file: UploadFile = File(...), granularity: str = Form("Par mois")):
    df = pd.read_csv(file.file)

    df['articleCreatedDate'] = pd.to_datetime(df['articleCreatedDate'], errors='coerce')
    df['sentimentHumanReadable'] = df['sentimentHumanReadable'].astype(str).str.strip().str.lower()
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
    top_authors_sentiment = author_sentiment.sort_values(by='Total', ascending=False).head(10)
    top_authors_sentiment = top_authors_sentiment.drop(columns='Total')
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

    html_report = f"""<!DOCTYPE html>
    <html lang='fr'>
    <head><meta charset='UTF-8'><title>Rapport de Veille Médiatique</title></head>
    <body>
        <h1>📊 Rapport de Veille Médiatique</h1>
        <ul>
            <li><strong>Mentions totales :</strong> {kpis['total_mentions']}</li>
            <li><strong>Positives :</strong> {kpis['positive']}</li>
            <li><strong>Négatives :</strong> {kpis['negative']}</li>
            <li><strong>Neutres :</strong> {kpis['neutral']}</li>
        </ul>
        <img src="data:image/png;base64,{evolution_mentions_b64}" width="700"/><br/>
        <img src="data:image/png;base64,{sentiments_global_b64}" width="600"/><br/>
        <img src="data:image/png;base64,{sentiments_auteurs_b64}" width="700"/><br/>
        <p style="font-size: 12px; color: #999;">Généré automatiquement avec FastAPI</p>
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
