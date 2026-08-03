import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import zipfile
import os

# ==========================================
# KONFIGURASI HALAMAN & CUSTOM CSS (AESTHETIC)
# ==========================================
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { color: #4A4036; font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    .stButton>button { background-color: #C1A58D; color: white; border-radius: 8px; border: none; padding: 10px 24px; transition: 0.3s; }
    .stButton>button:hover { background-color: #A68B75; }
    .css-1d391kg { background-color: #F3EFEA; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DEFINISI KELAS KERAS
# ==========================================
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_book_title, embedding_size, dropout_rate=0.2, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_title_embedding = layers.Embedding(
            num_book_title, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.book_title_bias = layers.Embedding(num_book_title, 1)
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_vector = self.dropout(user_vector)
        user_bias = self.user_bias(inputs[:, 0])
        
        book_title_vector = self.book_title_embedding(inputs[:, 1])
        book_title_vector = self.dropout(book_title_vector)
        book_title_bias = self.book_title_bias(inputs[:, 1])
        
        dot_user_book_title = tf.tensordot(user_vector, book_title_vector, 2)
        x = dot_user_book_title + user_bias + book_title_bias
        return tf.nn.sigmoid(x)

# ==========================================
# FUNGSI LOAD DATA (Menggunakan Cache agar Cepat)
# ==========================================
@st.cache_data
def load_data():
    books = pd.read_csv('books_clean.csv')
    
    # 1. Deteksi nama file zip yang ada di GitHub
    zip_file_path = None
    if os.path.exists('cosine_sim.zip'):
        zip_file_path = 'cosine_sim.zip'
    elif os.path.exists('cosine_sim.pkl.zip'):
        zip_file_path = 'cosine_sim.pkl.zip'
        
    # 2. Ekstrak cerdas (mengabaikan folder di dalam zip)
    if not os.path.exists('cosine_sim.pkl') and zip_file_path:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Cari file apa saja di dalam zip yang berakhiran .pkl
                if file_info.filename.endswith('.pkl'):
                    # Paksa ekstrak ke folder utama dengan nama 'cosine_sim.pkl'
                    file_info.filename = 'cosine_sim.pkl' 
                    zip_ref.extract(file_info, '.')
                    break # Berhenti setelah menemukan file pkl
            
    # 3. Load file yang sudah diekstrak
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
        
    with open('mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        
    return books, cosine_sim, mappings

# ==========================================
# SIDEBAR NAVIGASI
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100)
st.sidebar.title("Navigasi Menu")
menu = st.sidebar.radio("Pilih Fitur:", ["Beranda", "Cari Buku Serupa (Content-Based)", "Rekomendasi Personal (Collaborative)"])

# ==========================================
# MENU 1: BERANDA
# ==========================================
if menu == "Beranda":
    st.title("📚 Sistem Rekomendasi Buku")
    st.write("Selamat datang di prototipe sistem rekomendasi buku. Silakan gunakan menu di sebelah kiri untuk mengeksplorasi rekomendasi berdasarkan kemiripan judul atau preferensi pengguna.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Buku", f"{len(books):,}")
    with col2:
        st.metric("Total Penulis", f"{len(books['book_author'].unique()):,}")
    with col3:
        st.metric("Total Pengguna Aktif", f"{num_users:,}")
        
    st.markdown("---")
    st.subheader("Sekilas Data Buku")
    st.dataframe(books[['isbn', 'book_title', 'book_author', 'year_of_publication']].head(15), use_container_width=True)

# ==========================================
# MENU 2: CONTENT-BASED FILTERING
# ==========================================
elif menu == "Cari Buku Serupa (Content-Based)":
    st.title("🔍 Temukan Buku Serupa")
    st.write("Sistem akan mencari buku dengan penulis dan karakteristik yang mirip dengan buku pilihan Anda.")
    
    # Dropdown judul buku
    book_list = books['book_title'].unique()
    selected_book = st.selectbox("Pilih judul buku yang Anda sukai:", book_list)
    
    if st.button("Cari Rekomendasi"):
        with st.spinner('Mencari buku terbaik untuk Anda...'):
            try:
                index = cosine_sim_df.loc[:, selected_book].to_numpy().argpartition(range(-1, -6, -1))
                closest = cosine_sim_df.columns[index[-1:-(5+2):-1]]
                closest = closest.drop(selected_book, errors='ignore')
                
                # Bungkus array ke dalam dictionary Pandas
                result_df = pd.DataFrame({'book_title': closest})
                result_df = result_df.merge(books[['book_title', 'book_author']], on='book_title')
                result_df = result_df.rename(columns={'book_title': 'Judul Buku', 'book_author': 'Penulis'}).drop_duplicates().head(5)
                
                st.success("Berhasil menemukan buku serupa!")
                st.table(result_df)
            except KeyError:
                st.error("Buku tidak ditemukan dalam matriks similarity.")

# ==========================================
# MENU 3: COLLABORATIVE FILTERING
# ==========================================
elif menu == "Rekomendasi Personal (Collaborative)":
    st.title("✨ Rekomendasi Personal")
    st.write("Prediksi buku yang mungkin akan Anda beri rating tinggi berdasarkan riwayat bacaan Anda.")
    
    user_list = list(mappings['user_to_user_encoded'].keys())
    selected_user = st.selectbox("Pilih User ID Anda:", user_list[:100])
    
    if st.button("Tampilkan Rekomendasi"):
        if model is None:
            st.error("Model gagal dimuat. Pastikan Anda sudah mengunduh file 'model_weights.pkl' dari Colab.")
        else:
            with st.spinner('Memproses pola bacaan Anda...'):
                user_encoder = mappings['user_to_user_encoded'].get(selected_user)
                all_isbn = list(mappings['isbn_to_isbn_encoded'].keys())
                
                # Format input untuk Keras Model
                book_not_readed_encoded = [mappings['isbn_to_isbn_encoded'][x] for x in all_isbn]
                user_book_array = np.column_stack((
                    np.full(len(book_not_readed_encoded), user_encoder), 
                    book_not_readed_encoded
                ))
                
                # Prediksi Keras
                ratings_model = model.predict(user_book_array, verbose=0).flatten()
                top_ratings_indices = ratings_model.argsort()[-10:][::-1]
                
                recommended_isbn = [all_isbn[i] for i in top_ratings_indices]
                recommended_books = books[books['isbn'].isin(recommended_isbn)][['book_title', 'book_author']].drop_duplicates()
                recommended_books = recommended_books.rename(columns={'book_title': 'Judul Buku', 'book_author': 'Penulis'})
                
                st.success(f"Top 10 Rekomendasi untuk User: {selected_user}")
                st.table(recommended_books.reset_index(drop=True))
