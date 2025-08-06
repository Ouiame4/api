from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

palette_custom = ["#81C3D7", "#219ebc", "#D9DCD6", "#2F6690", "#16425B"]

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.post("/analyser_json")
async def analyser_json(request: Request):
    try:
        json_data = await request.json()
        if not isinstance(json_data, list):
            return {"error": "Invalid data format. Expected a list of articles."}

        df = pd.DataFrame(json_data)

        # Convert timestamp to datetime
        df['published_at'] = pd.to_datetime(df['published_at'], unit='s')
        df = df.rename(columns={
            'published_at': 'articleCreatedDate',
            'author': 'authorName',
            'sentiment_label': 'sentimentHumanReadable'
        })

        df['Year'] = df['articleCreatedDate'].dt.year
        df['Period'] = df['articleCreatedDate'].dt.to_period('M')

        # KPI
        kpis = {
            "total_mentions": int(df.shape[0]),
            "positive": int((df['sentimentHumanReadable'] == 'positive').sum()),
            "negative": int((df['sentimentHumanReadable'] == 'negative').sum()),
            "neutral": int((df['sentimentHumanReadable'] == 'neutral').sum()),
        }

        # Graph 1: Mentions over time
        mentions_over_time = df['Period'].value_counts().sort_index()
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(mentions_over_time.index.astype(str), mentions_over_time.values, marker='o', linestyle='-', color="#2F6690")
        ax1.set_title("Évolution des mentions")
        ax1.set_xlabel("Période")
        ax1.set_ylabel("Mentions")
        plt.xticks(rotation=45)
        graph1_b64 = fig_to_base64(fig1)
        plt.close(fig1)

        # Graph 2: Sentiment distribution
        sentiment_counts = df['sentimentHumanReadable'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=palette_custom)
        ax2.set_title("Répartition globale des sentiments")
        ax2.set_ylabel("Nombre d'articles")
        graph2_b64 = fig_to_base64(fig2)
        plt.close(fig2)

        # Graph 3: Sentiment by author
        author_sentiment = df.groupby(['authorName', 'sentimentHumanReadable']).size().unstack(fill_value=0)
        author_sentiment['Total'] = author_sentiment.sum(axis=1)
        top_authors = author_sentiment.sort_values('Total', ascending=False).head(10).drop(columns='Total')
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        top_authors.plot(kind='barh', stacked=True, ax=ax3)
        ax3.set_title("Répartition des sentiments par auteur")
        graph3_b64 = fig_to_base64(fig3)
        plt.close(fig3)

        # Table
        top_authors_table = (
            df['authorName'].value_counts()
            .reset_index()
            .rename(columns={'index': 'Auteur', 'authorName': 'Nombre d’articles'})
            .head(10)
            .to_html(index=False, border=1)
        )

        html_report = f"""
        <h1>📊 Rapport d'Analyse de Veille Médiatique</h1>
        <p><strong>Total mentions:</strong> {kpis['total_mentions']}</p>
        <p><strong>Positives:</strong> {kpis['positive']}</p>
        <p><strong>Négatives:</strong> {kpis['negative']}</p>
        <p><strong>Neutres:</strong> {kpis['neutral']}</p>

        <div><h2>Évolution des mentions</h2><img src="data:image/png;base64,{graph1_b64}"/></div>
        <div><h2>Répartition globale des sentiments</h2><img src="data:image/png;base64,{graph2_b64}"/></div>
        <div><h2>Répartition des sentiments par auteur</h2><img src="data:image/png;base64,{graph3_b64}"/></div>
        <div><h2>Top auteurs</h2>{top_authors_table}</div>
        """

        return {
            "html_report": html_report,
            "kpis": kpis
        }

    except Exception as e:
        return {"error": str(e)}
