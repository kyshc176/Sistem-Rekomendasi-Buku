import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
import os

# ==========================================
# KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { color: #4A4036; font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    .stButton>button { background-color: #C1A58D; color: white; border-radius: 8px; border: none; padding: 10px 24px; transition: 0.3s; }
    .stButton>button:hover { background-color: #A68B75; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LAZY LOADERS (Fungsi dipisah agar memori aman)
# ==========================================
@st.cache_data
def load_books():
    if not os.path.exists('books_clean.csv'):
        for csv_zip in ['csv.zip', 'books_clean.zip', 'books_clean.csv.zip']:
            if os.path.exists(csv_zip):
                with zipfile.ZipFile(csv_zip, 'r') as zip_ref:
                    for file_info in zip_ref.infolist():
                        if file_info.filename.endswith('.csv'):
                            file_info.filename = 'books_clean.csv'
                            zip_ref.extract(file_info, '.')
                            break
                break
    return pd.read_csv('books_clean.csv')

@st.cache_data
def load_content_based_data():
    zip_file_path = None
    if os.path.exists('cosine_sim.zip'):
        zip_file_path = 'cosine_sim.zip'
    elif os.path.exists('cosine_sim.pkl.zip'):
        zip_file_path = 'cosine_sim.pkl.zip'
        
    if not os.path.exists('cosine_sim.pkl') and zip_file_path:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.pkl'):
                    file_info.filename = 'cosine_sim.pkl' 
                    zip_ref.extract(file_info, '.')
                    break
            
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    return cosine_sim

@st.cache_data
def load_mappings():
    with open('mappings.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_collab_model(num_users, num_book_title):
    # Import TensorFlow HANYA saat model dipanggil
    import tensorflow as tf
    from tensorflow.keras import layers

    class RecommenderNet(tf.keras.Model):
        def __init__(self, num_users, num_book_title, embedding_size, dropout_rate=0.2, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal')
            self.user_bias = layers.Embedding(num_users, 1)
            self.book_title_embedding = layers.Embedding(num_book_title, embedding_size, embeddings_initializer='he_normal')
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

    model = RecommenderNet(num_users, num_book_title, 50)
    model(tf.constant([[0, 0]])) # Pancing model
    
    with open('model_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    return model

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
    st.write("Selamat datang! Sistem ini menggunakan Machine Learning untuk merekomendasikan buku.")
    
    with st.spinner("Mengekstrak dan memuat dataset..."):
        try:
            books = load_books()
        except Exception as e:
            st.error(f"Gagal memuat dataset buku: {e}")
            st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Buku", f"{len(books):,}")
    with col2:
        st.metric("Total Penulis", f"{len(books['book_author'].unique()):,}")
        
    st.markdown("---")
    st.subheader("Sekilas Data Buku")
    # Menggunakan width='stretch' sesuai standar terbaru Streamlit
    st.dataframe(books[['isbn', 'book_title', 'book_author', 'year_of_publication']].head(15), width='stretch')

# ==========================================
# MENU 2: CONTENT-BASED FILTERING
# ==========================================
elif menu == "Cari Buku Serupa (Content-Based)":
    st.title("🔍 Temukan Buku Serupa")
    
    with st.spinner('Menyiapkan AI Pencari Kemiripan (Proses ini mungkin memakan waktu sebentar)...'):
        try:
            books = load_books()
            cosine_sim_df = load_content_based_data()
        except Exception as e:
            st.error(f"Gagal memuat model atau data: {e}")
            st.stop()

    book_list = books['book_title'].unique()
    selected_book = st.selectbox("Pilih judul buku yang Anda sukai:", book_list)
    
    if st.button("Cari Rekomendasi"):
        try:
            # Cek apakah bentuknya numpy array (sering terjadi dari ekspor Scikit-Learn)
            if isinstance(cosine_sim_df, np.ndarray):
                # Cari index angka dari buku yang dipilih
                idx = books[books['book_title'] == selected_book].index[0]
                
                # Ambil skor kemiripan, lalu urutkan dari yang terbesar ke terkecil
                sim_scores = list(enumerate(cosine_sim_df[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                # Ambil 5 buku teratas (index 1 ke 6 karena index 0 biasanya adalah buku itu sendiri)
                sim_scores = sim_scores[1:6] 
                book_indices = [i[0] for i in sim_scores]
                
                # Dapatkan data buku berdasarkan index
                result_df = books.iloc[book_indices][['book_title', 'book_author']].drop_duplicates().head(5)
                result_df = result_df.rename(columns={'book_title': 'Judul Buku', 'book_author': 'Penulis'})
                
            else:
                # Logika jika bentuknya Pandas DataFrame (memiliki nama kolom judul buku)
                index = cosine_sim_df.loc[:, selected_book].to_numpy().argpartition(range(-1, -6, -1))
                closest = cosine_sim_df.columns[index[-1:-(5+2):-1]]
                closest = closest.drop(selected_book, errors='ignore')
                
                result_df = pd.DataFrame({'book_title': closest}).merge(books[['book_title', 'book_author']], on='book_title').drop_duplicates().head(5)
                result_df = result_df.rename(columns={'book_title': 'Judul Buku', 'book_author': 'Penulis'})
            
            st.success("Berhasil menemukan buku serupa!")
            st.table(result_df.reset_index(drop=True))
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mencari buku: {e}")
            st.info("Saran: Pastikan judul buku yang dipilih memiliki data yang sinkron dengan matriks kemiripan.")

# ==========================================
# MENU 3: COLLABORATIVE FILTERING
# ==========================================
elif menu == "Rekomendasi Personal (Collaborative)":
    st.title("✨ Rekomendasi Personal")
    
    with st.spinner('Menyiapkan Model Deep Learning TensorFlow...'):
        try:
            books = load_books()
            mappings = load_mappings()
            num_users = len(mappings['user_to_user_encoded'])
            num_book_title = len(mappings['isbn_to_isbn_encoded'])
            model = load_collab_model(num_users, num_book_title)
        except Exception as e:
            st.error(f"Gagal memuat AI Rekomendasi: {e}")
            st.stop()

    user_list = list(mappings['user_to_user_encoded'].keys())
    selected_user = st.selectbox("Pilih User ID Anda:", user_list[:100])
    
    if st.button("Tampilkan Rekomendasi"):
        with st.spinner('Memproses pola bacaan Anda...'):
            user_encoder = mappings['user_to_user_encoded'].get(selected_user)
            all_isbn = list(mappings['isbn_to_isbn_encoded'].keys())
            
            book_not_readed_encoded = [mappings['isbn_to_isbn_encoded'][x] for x in all_isbn]
            user_book_array = np.column_stack((
                np.full(len(book_not_readed_encoded), user_encoder), 
                book_not_readed_encoded
            ))
            
            ratings_model = model.predict(user_book_array, verbose=0).flatten()
            top_ratings_indices = ratings_model.argsort()[-10:][::-1]
            
            recommended_isbn = [all_isbn[i] for i in top_ratings_indices]
            recommended_books = books[books['isbn'].isin(recommended_isbn)][['book_title', 'book_author']].drop_duplicates()
            recommended_books = recommended_books.rename(columns={'book_title': 'Judul Buku', 'book_author': 'Penulis'})
            
            st.success(f"Top Rekomendasi untuk User: {selected_user}")
            st.table(recommended_books.reset_index(drop=True))
