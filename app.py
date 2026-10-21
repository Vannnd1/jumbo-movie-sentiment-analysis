import streamlit as st
import pickle
import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ============================================================
# 1. Konfigurasi Halaman Streamlit
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen Film Jumbo - SVM",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data jika belum ada di server Streamlit
for res in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

# ============================================================
# 2. Cache Resource (Muat Model, TF-IDF, Kamus Slang Sekali Saja)
# ============================================================
@st.cache_resource
def load_assets():
    # Muat Model SVM & TF-IDF
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # Muat Kamus Slang
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

    # Stopwords
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

    # Stemmer Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return model, tfidf, slang_dict, all_stopwords, stemmer, NEGATION_WORDS

model, tfidf, slang_dict, ALL_STOPWORDS, stemmer, NEGATION_WORDS = load_assets()

# ============================================================
# 3. Fungsi Preprocessing Teks
# ============================================================
def preprocess_text(text):
    # 1. Demojize
    text = emoji.demojize(str(text), language='en')
    text = re.sub(r':([a-zA-Z0-9_]+):', lambda m: ' ' + m.group(1).replace('_', ' ') + ' ', text)
    # 2. Lowercase & Clean Regex
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    # 3. Tokenize
    tokens = word_tokenize(text)
    # 4. Slang Normalization & rule imbuhan nge-...-in
    norm_tokens = []
    for w in tokens:
        if w in slang_dict:
            norm_tokens.append(slang_dict[w])
        elif re.match(r'^nge([a-z]+)in$', w):
            root = re.sub(r'^nge([a-z]+)in$', r'\1', w)
            norm_tokens.append(slang_dict.get(root, root))
        else:
            norm_tokens.append(w)
    # 5. Stopword removal (jaga kata sentimen & negasi)
    filtered = [w for w in norm_tokens if w not in ALL_STOPWORDS and len(w) > 1]
    # 6. Stemming
    stemmed = [stemmer.stem(w) for w in filtered]

    return ' '.join(stemmed), stemmed

# ============================================================
# 4. Sidebar Navigasi & Informasi Proyek
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/movie-projector.png", width=70)
    st.title("🎬 Film Jumbo")
    st.markdown("**Analisis Sentimen Komentar YouTube**")
    st.markdown("---")
    st.markdown("### 📌 Informasi Metodologi")
    st.markdown("""
    - **Algoritma**: Support Vector Machine (SVM)
    - **Kernel**: Linear (`class_weight='balanced'`)
    - **Pelabelan**: InSet Lexicon (Berbobot)
    - **Klasifikasi**: Biner (Positif vs Negatif)
    - **Akurasi Model**: **86.0%**
    - **F1-Score Positif**: **0.90**
    - **F1-Score Negatif**: **0.74**
    """)
    st.markdown("---")
    st.markdown("🔗 **[GitHub Repository](https://github.com/Vannnd1/jumbo-movie-sentiment-analysis)**")
    st.caption("Dibuat oleh: [@Vannnd1](https://github.com/Vannnd1)")

# ============================================================
# 5. Tampilan Utama Aplikasi
# ============================================================
st.title("🎬 Analisis Sentimen Penonton Film Animasi Jumbo")
st.markdown("""
Aplikasi web ini menggunakan **Machine Learning Support Vector Machine (SVM)** dan metode **InSet Lexicon** 
untuk memprediksi polaritas sentimen ulasan penonton terhadap teaser/trailer film animasi **Jumbo** di YouTube.
""")

tab1, tab2, tab3 = st.tabs([
    "🔍 Uji Sentimen Real-Time", 
    "📊 Dashboard Opini Penonton", 
    "📚 Tentang Penelitian"
])

# ------------------------------------------------------------
# TAB 1: Live Testing
# ------------------------------------------------------------
with tab1:
    st.subheader("Uji Kalimat Komentar Penonton")
    st.write("Ketikkan kalimat ulasan atau klik salah satu contoh di bawah ini:")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    sample_text = ""
    if col_btn1.button("Contoh 1 (Pujian Visual)"):
        sample_text = "Animasi jumbo gokil keren banget, visualnya mantap parah!"
    if col_btn2.button("Contoh 2 (Kritik Alur)"):
        sample_text = "Alurnya ngebosenin dan ceritanya hambar banget, kecewa."
    if col_btn3.button("Contoh 3 (Penyangkalan)"):
        sample_text = "Filmnya gak mengecewakan sama sekali, wajib nonton di bioskop!"
    if col_btn4.button("Contoh 4 (Kritik Halus)"):
        sample_text = "Kurang suka sama pengisi suaranya, masih agak kaku."

    user_input = st.text_area(
        "Masukkan teks komentar di sini:",
        value=sample_text,
        placeholder="Contoh: Keren banget visualnya bikin merinding...",
        height=100
    )

    if st.button("🚀 Analisis Sentimen Sekarang", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Mohon masukkan teks komentar terlebih dahulu.")
        else:
            with st.spinner("Memproses teks dan menganalisis sentimen..."):
                clean_text, tokens = preprocess_text(user_input)

                if not clean_text.strip():
                    st.info("ℹ️ Tidak ditemukan kata berbobot sentimen setelah pembersihan teks. Silakan coba kalimat yang lebih spesifik.")
                else:
                    vec = tfidf.transform([clean_text])
                    pred = model.predict(vec)[0]
                    decision_val = model.decision_function(vec)[0]

                    st.markdown("---")
                    st.subheader("Hasil Analisis:")

                    col_res1, col_res2 = st.columns([1, 2])

                    with col_res1:
                        if pred == "positive":
                            st.success("### 🟢 SENTIMEN POSITIF")
                            st.write("Penonton cenderung **menyukai** atau memberikan apresiasi.")
                        else:
                            st.error("### 🔴 SENTIMEN NEGATIF")
                            st.write("Penonton cenderung **mengkritik** atau merasa kurang puas.")

                    with col_res2:
                        with st.expander("🔎 Lihat Rincian Tahap Preprocessing Teks", expanded=True):
                            st.write(f"**Teks Asli**: {user_input}")
                            st.write(f"**Hasil Preprocessing**: `{clean_text}`")
                            st.write(f"**Token Kata**: `{tokens}`")
                            st.caption(f"Decision Function Score: {decision_val:.4f}")

# ------------------------------------------------------------
# TAB 2: Dashboard Opini Penonton
# ------------------------------------------------------------
with tab2:
    st.subheader("Statistik Data Opini Penonton Film Jumbo")
    st.write("Ringkasan distribusi polaritas dari ribuan komentar yang telah dianalisis pada trailer YouTube:")

    try:
        df_labeled = pd.read_excel("labeled_comments.xlsx")
        counts = df_labeled['sentiment'].value_counts()
        total_data = len(df_labeled)
        pos_count = counts.get('positive', 0)
        neg_count = counts.get('negative', 0)
        pos_pct = (pos_count / total_data) * 100
        neg_pct = (neg_count / total_data) * 100

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Komentar Dianalisis", f"{total_data:,}")
        col_m2.metric("Sentimen Positif", f"{pos_count:,} ({pos_pct:.1f}%)")
        col_m3.metric("Sentimen Negatif", f"{neg_count:,} ({neg_pct:.1f}%)")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.write("#### Distribusi Sentimen (Diagram Batang)")
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            ax_bar.bar(["Positif", "Negatif"], [pos_count, neg_count], color=["#28a745", "#dc3545"])
            ax_bar.set_ylabel("Jumlah Komentar")
            st.pyplot(fig_bar)

        with col_c2:
            st.write("#### Rasio Sentimen (Diagram Lingkaran)")
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            ax_pie.pie([pos_count, neg_count], labels=["Positif", "Negatif"], autopct="%1.1f%%", colors=["#28a745", "#dc3545"], startangle=90)
            st.pyplot(fig_pie)

    except Exception as e:
        st.info("Dataset 'labeled_comments.xlsx' belum dimuat untuk dashboard visual.")

# ------------------------------------------------------------
# TAB 3: Tentang Penelitian
# ------------------------------------------------------------
with tab3:
    st.subheader("Judul Resmi Penelitian:")
    st.info("**ANALISIS SENTIMEN KOMENTAR PADA PLATFORM YOUTUBE TENTANG FILM JUMBO MENGGUNAKAN ALGORITMA SUPPORT VECTOR MACHINE (SVM) DAN METODE INSET LEXICON**")

    st.markdown("""
    ### 🔬 Metodologi:
    1. **Pengumpulan Data**: Scraping komentar YouTube API v3 pada video teaser film animasi "Jumbo".
    2. **Prapemrosesan (NLP)**: Demojize emoji, normalisasi kata gaul (Kamus Alay), pelestarian kata negasi, stopword removal, dan stemming Sastrawi.
    3. **Pelabelan Sentimen**: InSet Lexicon berbobot (-5 s/d +5) yang disesuaikan untuk konteks ulasan film. Kelas netral disaring agar model fokus pada polaritas biner nyata.
    4. **Pembobotan Fitur**: TF-IDF N-gram (1,2) dengan `sublinear_tf=True`.
    5. **Klasifikasi**: Support Vector Machine (Linear Kernel) dengan `class_weight='balanced'` untuk menangani data tidak seimbang (*class imbalance*).
    """)
