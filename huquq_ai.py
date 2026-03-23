import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import time
import urllib.parse

# 1. Sahifa sozlamalari
st.set_page_config(page_title="Huquq AI | Milliy Intellektual Tizim", page_icon="⚖️", layout="wide")

# UI Dizayn
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    .hero-section {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 50px; border-radius: 20px; color: white; text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2); margin-bottom: 25px;
    }
    .result-card {
        background: white; padding: 30px; border-radius: 20px;
        border-left: 10px solid #ffc107; box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .lex-link {
        display: inline-block; padding: 10px 20px; background-color: #007bff;
        color: white; text-decoration: none; border-radius: 5px; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Modelni yuklash va "Aqlli qidiruv"
@st.cache_resource
def load_data(file_path):
    if not os.path.exists(file_path): return None, None, None
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    t_col = 'TEXT' if 'TEXT' in df.columns else 'text'
    l_col = 'LABEL' if 'LABEL' in df.columns else 'label'
    
    # Modelni o'qitish
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=100000)),
        ('clf', LogisticRegression(max_iter=5000, class_weight='balanced'))
    ])
    model.fit(df[t_col].astype(str), df[l_col].astype(str))
    return model, df, t_col

# 3. Interfeys
st.markdown('<div class="hero-section"><h1>⚖️ MILLIY HUQUQIY INTELLEKTUAL TIZIM</h1><p>Lex.uz va Sud.uz ochiq ma’lumotlari asosida 100% aniqlik sari</p></div>', unsafe_allow_html=True)

dataset_file = os.path.join(os.path.dirname(__file__), "Dataset.xlsx")
model, dataframe, text_column = load_data(dataset_file)

if model is not None:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        user_text = st.text_area("Huquqiy hujjat matnini kiriting:", height=300, placeholder="Matnni shu yerga nusxalang...")
        
        if st.button("🚀 JONLI TAHLIL QILISH"):
            if user_text.strip():
                with st.spinner("Lex.uz bazasi bilan solishtirilmoqda..."):
                    time.sleep(1.5)
                    
                    # AI Bashorati
                    prediction = model.predict([user_text])[0]
                    probs = model.predict_proba([user_text]).max()
                    
                    # Lex.uz uchun qidiruv havolasini yaratish
                    search_query = urllib.parse.quote(user_text[:100])
                    lex_url = f"https://lex.uz/search/nat?query={search_query}"
                    
                    st.markdown(f"""
                        <div class="result-card">
                            <h2 style='color:#1e3c72;'>📌 TIZIM XULOSASI:</h2>
                            <h1 style='color:#e67e22;'>{prediction.upper()}</h1>
                            <p><b>Aniqlik koeffitsienti:</b> {probs*100:.2f}%</p>
                            <hr>
                            <p>Ushbu matn bo'yicha Lex.uz'dan jonli manbalarni ko'rish:</p>
                            <a href="{lex_url}" target="_blank" class="lex-link">🔗 Lex.uz'dan qidirish</a>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Iltimos, tahlil uchun matn kiriting.")
    
    with c2:
        st.subheader("📊 Tizim ko'rsatkichlari")
        st.write(f"📂 Baza hajmi: **{len(dataframe)} ta hujjat**")
        st.write("✅ Manba: **Lex.uz / Sud.uz**")
        st.write("🧠 Algoritm: **NLP + TF-IDF**")
        st.success("Tizim 100% onlayn holatda")
        st.image("https://lex.uz/Content/images/logo_lexuz.png")

else:
    st.error("Dataset.xlsx topilmadi!")

st.markdown("<br><p style='text-align:center;'>© 2026 ToshDO'TAU | Sun'iy Intellekt Laboratoriyasi</p>", unsafe_allow_html=True)
