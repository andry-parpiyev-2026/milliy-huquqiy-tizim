import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import time

# 1. Sahifa dizayni va brending
st.set_page_config(
    page_title="Huquq AI | Milliy Intellektual Tizim",
    page_icon="⚖️",
    layout="wide"
)

# Professional UI dizayn
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .hero-section {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        padding: 40px; border-radius: 20px; color: white;
        text-align: center; margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .result-card {
        background-color: white; padding: 25px; border-radius: 15px;
        border-top: 8px solid #ffca28; box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: #1a237e; color: white; border-radius: 25px;
        height: 3.5em; width: 100%; font-weight: bold; border: none;
    }
    .stButton>button:hover { background: #0d47a1; color: #ffca28; }
    </style>
    """, unsafe_allow_html=True)

# 2. Modelni o'qitish (Lex.uz va Sud.uz terminologiyasi asosida)
@st.cache_resource
def train_legal_ai(file_path):
    if not os.path.exists(file_path):
        return None, None
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        # Ustunlarni aniqlash (TEXT va LABEL)
        t_col = 'TEXT' if 'TEXT' in df.columns else 'text'
        l_col = 'LABEL' if 'LABEL' in df.columns else 'label'
        
        df = df.dropna(subset=[t_col, l_col])
        
        # N-gram (1,3) - so'z birikmalarini tahlil qilish uchun
        model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=60000)),
            ('clf', LogisticRegression(max_iter=2500, class_weight='balanced'))
        ])
        
        model.fit(df[t_col].astype(str), df[l_col].astype(str))
        return model, len(df)
    except:
        return None, None

# 3. Sidebar
with st.sidebar:
    st.image("https://lex.uz/Content/images/logo_lexuz.png", width=180)
    st.markdown("---")
    st.info("Tizim Lex.uz va Sud.uz ochiq ma'lumotlari asosida ishlaydi.")
    if st.button("🔄 Modelni yangilash"):
        st.cache_resource.clear()
        st.rerun()

# 4. Asosiy Interfeys
st.markdown("""
    <div class="hero-section">
        <h1>⚖️ MILLIY HUQUQIY INTELLEKTUAL TIZIM</h1>
        <p>Sun'iy intellekt yordamida hujjatlarni avtomatik tasniflash</p>
    </div>
    """, unsafe_allow_html=True)

# Fayl yo'li (Kodingiz bilan bir xil papkada bo'lishi shart)
dataset_file = os.path.join(os.path.dirname(__file__), "Dataset.xlsx")

with st.spinner("⏳ Bazaviy bilimlar yuklanmoqda..."):
    model, row_count = train_legal_ai(dataset_file)

if model:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Tahlil uchun matn")
        user_input = st.text_area("Hujjat parchasini kiriting:", height=250, 
                                  placeholder="Masalan: Sud qarori ijrosini ta'minlash tartibi...")
        
        if st.button("🔍 CHUQUR TAHLIL QILISH"):
            if user_input.strip():
                with st.spinner("AI matnni tahlil qilmoqda..."):
                    time.sleep(1) # Vizual effekt
                    pred = model.predict([user_input])[0]
                    prob = model.predict_proba([user_input]).max()
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h3 style='color:#1a237e;'>📋 TAHLIL NATIJASI:</h3>
                        <h1 style='text-align:center; color:#0d47a1;'>{pred.upper()}</h1>
                        <hr>
                        <p style='text-align:right;'><b>Ishonch darajasi:</b> {prob*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Iltimos, matn kiriting!")

    with col2:
        st.subheader("📊 Statistika")
        st.metric("O'qitilgan hujjatlar", f"{row_count:,}")
        st.metric("Tizim holati", "Onlayn")
        st.image("https://sud.uz/wp-content/uploads/2020/06/logo-new-uz.png", width=150)

else:
    st.error(f"❌ Fayl topilmadi: 'Dataset.xlsx' fayli huquq_ai.py bilan bir xil papkada ekanligini tekshiring.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>© 2026 ToshDO'TAU | Sud.uz bilan hamkorlikda</p>", unsafe_allow_html=True)
