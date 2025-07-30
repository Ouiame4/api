from fastapi import FastAPI, UploadFile, File, Form 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
from preprocessing import clean_dataframe  

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

@app.post("/analyser")
async def analyser_csv(file: UploadFile = File(...), granularity: str = Form("Par mois")):
    df = clean_dataframe(pd.read_csv(file.file))
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
        .rename(columns={'index': 'count', 'authorName': 'Auteur'})
        .head(10)
        .to_html(index=False, border=1, classes="styled-table")
    )

    html_report = f"""<!DOCTYPE html>
<html lang='fr'>
<head>
    <meta charset='UTF-8'>
    <title>Rapport de Veille Médiatique</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
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
        p, ul {{
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
        .interpretation {{
            font-style: italic;
            color: #444;
            margin: 10px auto 40px;
            max-width: 800px;
        }}
        .btn-download {{
            text-align: right;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

    <div class="btn-download">
        <button onclick="downloadPDF()" style="padding: 8px 16px; background-color: #2F6690; color: white; border: none; border-radius: 5px; cursor: pointer;">
            📄 Télécharger en PDF
        </button>
    </div>

    <h1>📊 Rapport d'Analyse de Veille Médiatique</h1>

    <p>
        Ce rapport de veille médiatique présente une analyse approfondie des articles publiés autour d’un sujet d’actualité.
        Il a pour objectif de fournir aux décideurs une vision claire et synthétique des mentions médiatiques, des perceptions exprimées
        (positives, négatives ou neutres) ainsi que des sources les plus influentes.
    </p>

    <h2>Indicateurs Clés</h2>
    <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        <div style="text-align: center;">
            <h3 style="margin-bottom: 5px;">{kpis['total_mentions']}</h3>
            <p style="margin: 0;">Mentions totales</p>
        </div>
        <div style="text-align: center;">
            <h3 style="margin-bottom: 5px;">{kpis['positive']}</h3>
            <p style="margin: 0;">Positives</p>
        </div>
        <div style="text-align: center;">
            <h3 style="margin-bottom: 5px;">{kpis['negative']}</h3>
            <p style="margin: 0;">Négatives</p>
        </div>
        <div style="text-align: center;">
            <h3 style="margin-bottom: 5px;">{kpis['neutral']}</h3>
            <p style="margin: 0;">Neutres</p>
        </div>
    </div>

    <div class="image-block">
        <h2>Évolution des mentions</h2>
        <img src="data:image/png;base64,{evolution_mentions_b64}" width="700"/>
        <p class="interpretation">
            Le nombre de mentions reste très faible jusqu’en mars 2025, puis connaît une forte hausse à partir de mai,
            signe d’un intérêt soudain ou d’un événement marquant.
        </p>
    </div>

    <div class="image-block">
        <h2>Répartition globale des sentiments</h2>
        <img src="data:image/png;base64,{sentiments_global_b64}" width="600"/>
        <p class="interpretation">
            La plupart des articles sont positifs, ce qui montre une image globalement favorable.
            Les articles négatifs restent peu nombreux.
        </p>
    </div>

    <div class="image-block">
        <h2>Répartition des sentiments par auteur</h2>
        <img src="data:image/png;base64,{sentiments_auteurs_b64}" width="700"/>
        <p class="interpretation">
            Les auteurs les plus actifs sont <em>'news-webmaster@google.com'</em>, <em>'Fox News'</em> et <em>'Inconnu'</em>.
            Les articles de <em>'news-webmaster@google.com'</em> sont majoritairement positifs.
            <em>'Fox News'</em> présente une répartition plus variée, avec une part importante d’articles négatifs.
            Les articles d’auteur <em>'Inconnu'</em> sont surtout neutres.Globalement, les sentiments dominants dans le corpus sont positifs et neutres.
        </p>
    </div>

    <h2>Top 10 Auteurs les plus actifs</h2>
    {top_table}

    <script>
        function downloadPDF() {{
            const element = document.body;
            const opt = {{
                margin:       0.5,
                filename:     'rapport_veille.pdf',
                image:        {{ type: 'jpeg', quality: 0.98 }},
                html2canvas:  {{ scale: 2 }},
                jsPDF:        {{ unit: 'in', format: 'a4', orientation: 'portrait' }}
            }};
            html2pdf().set(opt).from(element).save();
        }}
    </script>

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

@app.get("/rapport")
def get_rapport():
    return FileResponse("static/rapport_veille.html", media_type="text/html")
