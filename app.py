import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ============================================================
# 1. Konfigurasi Halaman & Styling Minimalis Profesional
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen: Film Jumbo (SVM)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan elegan, clean, dan bebas "AI Slop"
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header typography */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Clean metric card */
    .metric-container {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        background-color: #ffffff;
    }
    
    /* Status Pills */
    .pill-positive {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    .pill-negative {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Clean divider */
    hr {
        margin: 2rem 0;
        border-color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# Resource NLTK
for res in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

# ============================================================
# 2. Muat Model, Vectorizer & Kamus Terlatih
# ============================================================
@st.cache_resource
def load_model_assets():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    try:
        slang_df = pd.read_csv("kamus_alay.csv")
        slang_dict = dict(zip(
            slang_df['slang'].astype(str).str.lower().str.strip(),
            slang_df['formal'].astype(str).str.lower().str.strip()
        ))
    except Exception:
        slang_dict = {}

    custom_slang = {
        'ngebosenin': 'bosan',
        'bosenin': 'bosan',
        'bosen': 'bosan',
        'ngeselin': 'kesal',
        'ngecewain': 'kecewa',
        'kecewa': 'kecewa',
        'gak': 'tidak',
        'ga': 'tidak',
        'nggak': 'tidak',
        'ngga': 'tidak',
        'g': 'tidak',
        'gk': 'tidak',
        'bgt': 'sangat',
        'banget': 'sangat',
        'bgtt': 'sangat',
        'keren': 'bagus',
        'mantul': 'bagus',
        'mantap': 'bagus',
        'jelek': 'buruk',
        'ancur': 'buruk',
        'hancur': 'buruk'
    }
    slang_dict.update(custom_slang)

    NEGATION_WORDS = {
        "tidak", "bukan", "jangan", "kurang", "belum", "tanpa",
        "anti", "gak", "ga", "nggak", "ndak", "tak", "enggak",
        "kagak", "nda", "tiada"
    }
    DOMAIN_STOPWORDS = {"jumbo", "trailer", "video", "ymqdgbzmbdk"}
    PROTECTED_WORDS = {"tidak", "bukan", "kurang", "jangan", "bagus", "buruk", "bosan", "keren", "jelek"}

    base_sw = set(stopwords.words('indonesian'))
    try:
        with open('stopword.txt', 'r', encoding='utf-8') as f:
            custom_sw = set(line.strip().lower() for line in f if line.strip())
    except Exception:
        custom_sw = set()

    all_stopwords = (base_sw | custom_sw | DOMAIN_STOPWORDS) - NEGATION_WORDS - PROTECTED_WORDS

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return model, tfidf, slang_dict, all_stopwords, stemmer, NEGATION_WORDS

model, tfidf, slang_dict, ALL_STOPWORDS, stemmer, NEGATION_WORDS = load_model_assets()

# ============================================================
# 3. Pipeline Preprocessing Teks
# ============================================================
def preprocess(text):
    # Demojize internal
    text = emoji.demojize(str(text), language='en')
    text = re.sub(r':([a-zA-Z0-9_]+):', lambda m: ' ' + m.group(1).replace('_', ' ') + ' ', text)
    # Cleaning
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    # Tokenisasi
    tokens = word_tokenize(text)
    # Normalisasi slang
    norm_tokens = []
    for w in tokens:
        if w in slang_dict:
            norm_tokens.append(slang_dict[w])
        elif re.match(r'^nge([a-z]+)in$', w):
            root = re.sub(r'^nge([a-z]+)in$', r'\1', w)
            norm_tokens.append(slang_dict.get(root, root))
        else:
            norm_tokens.append(w)
    # Filter kata tidak bermakna
    filtered = [w for w in norm_tokens if w not in ALL_STOPWORDS and len(w) > 1]
    # Stemming Sastrawi
    stemmed = [stemmer.stem(w) for w in filtered]

    return ' '.join(stemmed), stemmed

# ============================================================
# 4. Header & Pengantar Akademik
# ============================================================
st.title("Analisis Sentimen Komentar YouTube: Film Jumbo")
st.markdown(
    "Implementasi Model **Support Vector Machine (SVM)** dengan Pelabelan **InSet Lexicon** "
    "untuk Klasifikasi Sentimen Biner Opini Penonton."
)

st.markdown("---")

# ============================================================
# 5. Navigasi Tab
# ============================================================
tab_infer, tab_data, tab_model = st.tabs([
    "Pengujian Model", 
    "Distribusi Korpus", 
    "Spesifikasi Teknis"
])

# ------------------------------------------------------------
# TAB 1: Pengujian Model (Inference)
# ------------------------------------------------------------
with tab_infer:
    st.markdown("### Pengujian Inferensi Teks")
    st.write("Masukkan ulasan komentar penonton untuk menguji respon klasifikasi model secara langsung.")

    col_input, col_meta = st.columns([2, 1])

    with col_input:
        preset = st.selectbox(
            "Pilih sampel pengujian (opsional):",
            [
                "Ketik ulasan sendiri...",
                "Animasi film Jumbo keren banget, kualitas visualnya luar biasa.",
                "Alurnya membosankan dan ceritanya hambar, sangat mengecewakan.",
                "Karya anak bangsa yang patut diapresiasi, bangga banget!",
                "Pengisi suaranya masih kaku dan animasi gerakannya kurang halus."
            ]
        )

        default_text = "" if preset == "Ketik ulasan sendiri..." else preset
        user_text = st.text_area(
            "Teks Komentar:",
            value=default_text,
            height=110,
            placeholder="Tuliskan ulasan komentar di sini..."
        )

        submit = st.button("Jalankan Klasifikasi", type="primary")

    with col_meta:
        st.markdown("**Parameter Inferensi:**")
        st.markdown("""
        - **Model**: Linear SVM (`C=1.0`)
        - **Bobot Kelas**: `balanced`
        - **Ekstraksi**: TF-IDF Unigram + Bigram
        - **Kamus Fitur**: 1.644 n-grams
        """)

    if submit and user_text.strip():
        clean_text, tokens = preprocess(user_text)

        if not clean_text.strip():
            st.warning("Teks tidak mengandung token yang dapat dikenali setelah preprocessing.")
        else:
            vec = tfidf.transform([clean_text])
            prediction = model.predict(vec)[0]
            decision = model.decision_function(vec)[0]

            st.markdown("---")
            st.markdown("### Hasil Evaluasi Sentimen")

            col_res_a, col_res_b, col_res_c = st.columns([1, 1, 2])

            with col_res_a:
                st.markdown("**Prediksi Kelas:**")
                if prediction == "positive":
                    st.markdown('<span class="pill-positive">POSITIF</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="pill-negative">NEGATIF</span>', unsafe_allow_html=True)

            with col_res_b:
                st.markdown("**Decision Margin:**")
                st.metric(
                    label="Margin Jarak (w·x + b)",
                    value=f"{decision:+.4f}"
                )

            with col_res_c:
                st.markdown("**Representasi Fitur Bersih:**")
                st.code(clean_text, language="text")

            # Analisis Fitur Kata Kunci yang Berkontribusi
            feature_names = tfidf.get_feature_names_out()
            feature_indices = vec.nonzero()[1]
            coefs = model.coef_[0]

            matched_features = []
            for idx in feature_indices:
                word = feature_names[idx]
                weight = coefs[idx]
                matched_features.append({
                    "Fitur N-Gram": word,
                    "Bobot SVM": round(float(weight), 4),
                    "Pengaruh": "Positif" if weight > 0 else "Negatif"
                })

            if matched_features:
                st.markdown("**Dekomposisi Bobot Fitur pada Kalimat:**")
                df_features = pd.DataFrame(matched_features).sort_values(by="Bobot SVM", ascending=False)
                st.dataframe(df_features, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 2: Distribusi Korpus & Data
# ------------------------------------------------------------
with tab_data:
    st.markdown("### Ringkasan Korpus Data Latih")
    
    try:
        df_corpus = pd.read_excel("labeled_comments.xlsx")
        n_total = len(df_corpus)
        counts = df_corpus['sentiment'].value_counts()
        n_pos = counts.get('positive', 0)
        n_neg = counts.get('negative', 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Komentar Biner", f"{n_total:,}")
        m2.metric("Sentimen Positif", f"{n_pos:,}", f"{(n_pos/n_total)*100:.1f}%")
        m3.metric("Sentimen Negatif", f"{n_neg:,}", f"{(n_neg/n_total)*100:.1f}%")

        st.markdown("---")
        c_left, c_right = st.subplots if hasattr(st, "subplots") else (None, None)

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("**Distribusi Proporsi Kelas**")
            fig_bar, ax_bar = plt.subplots(figsize=(5, 3.5), facecolor='#ffffff')
            bars = ax_bar.bar(
                ["Positif", "Negatif"], 
                [n_pos, n_neg], 
                color=["#2b6cb0", "#c53030"],
                width=0.45
            )
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['left'].set_color('#cbd5e1')
            ax_bar.spines['bottom'].set_color('#cbd5e1')
            ax_bar.tick_params(colors='#475569')
            for bar in bars:
                h = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2, h + 15, f"{int(h):,}", ha='center', color='#334155', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

        with col_g2:
            st.markdown("**WordCloud Kata Dominan**")
            all_words = df_corpus['comment_clean'].astype(str).str.cat(sep=' ')
            wc = WordCloud(
                width=600, 
                height=420, 
                background_color='white', 
                colormap='bone',
                max_words=60
            ).generate(all_words)
            fig_wc, ax_wc = plt.subplots(figsize=(5, 3.5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            plt.tight_layout()
            st.pyplot(fig_wc)
            plt.close()

    except Exception as e:
        st.caption("File korpus 'labeled_comments.xlsx' tidak ditemukan.")

# ------------------------------------------------------------
# TAB 3: Spesifikasi Teknis & Evaluasi
# ------------------------------------------------------------
with tab_model:
    st.markdown("### Ringkasan Spesifikasi Model")
    
    st.markdown("""
    | Komponen | Spesifikasi Teknis |
    | :--- | :--- |
    | **Sumber Data** | Komentar YouTube Data API v3 (`video_id: yMqDgbZmBdk`) |
    | **Metode Pelabelan** | InSet Lexicon Berbobot (-5 s/d +5) dengan penyesuaian domain ulasan film |
    | **Skema Klasifikasi** | Biner (Sentimen Positif vs Sentimen Negatif) |
    | **Representasi Fitur** | TF-IDF Vectorizer (Unigram & Bigram, `sublinear_tf=True`, `min_df=2`) |
    | **Algoritma Utama** | Support Vector Classification (`kernel='linear'`, `class_weight='balanced'`) |
    | **Validasi Model** | 5-Fold Stratified Cross-Validation |
    """)

    st.markdown("---")
    st.markdown("### Kinerja Model pada Data Uji (20% Test Split)")

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        report_data = {
            "Kelas": ["Negatif", "Positif", "Akurasi Rata-rata"],
            "Precision": ["0.71", "0.92", "-"],
            "Recall": ["0.77", "0.89", "-"],
            "F1-Score": ["0.74", "0.90", "0.86"],
            "Dukungan Uji": [61, 175, 236]
        }
        st.dataframe(pd.DataFrame(report_data), hide_index=True, use_container_width=True)

    with col_t2:
        st.markdown("""
        **Catatan Metrik:**
        - Akurasi pengujian mencapai **86.0%**.
        - Bobot kelas seimbang (*balanced weighting*) berhasil menjaga *Recall* kelas negatif pada angka **77.0%** tanpa degradasi presisi.
        """)
