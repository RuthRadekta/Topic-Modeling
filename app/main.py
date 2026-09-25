import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Konfigurasi Halaman (Harus di awal)
st.set_page_config(page_title="Topic Modeling Dashboard", page_icon="📊", layout="wide")

# 2. Fungsi untuk memuat data
@st.cache_data
def load_raw_data():
    try:
        df_bertopic = pd.read_csv("data/evaluasi/metrik_bertopic.csv")
        df_lda = pd.read_csv("data/evaluasi/metrik_lda.csv")
        df_sss = pd.read_csv("data/evaluasi/metrik_s3.csv")
        
        df_bertopic['Model'] = 'BERTopic'
        df_lda['Model'] = 'LDA'
        df_sss['Model'] = 'Semantic Signal Separation'
        
        return pd.concat([df_bertopic, df_lda, df_sss], ignore_index=True)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_stats_data():
    try:
        df_stats_bertopic = pd.read_csv("data/evaluasi/stats_bertopic.csv")
        df_stats_lda = pd.read_csv("data/evaluasi/stats_lda.csv")
        df_stats_sss = pd.read_csv("data/evaluasi/stats_s3.csv")
        
        df_stats_bertopic['Model'] = 'BERTopic'
        df_stats_lda['Model'] = 'LDA'
        df_stats_sss['Model'] = 'Semantic Signal Separation'
        
        return pd.concat([df_stats_bertopic, df_stats_lda, df_stats_sss], ignore_index=True)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_topic_data():
    topik_dict = {}
    try:
        topik_dict['BERTopic'] = pd.read_csv("data/topik/topik_bertopic.csv")
    except FileNotFoundError:
        topik_dict['BERTopic'] = pd.DataFrame()
    try:
        topik_dict['LDA'] = pd.read_csv("data/topik/topik_lda.csv")
    except FileNotFoundError:
        topik_dict['LDA'] = pd.DataFrame()
    try:
        topik_dict['Semantic Signal Separation'] = pd.read_csv("data/topik/topik_s3.csv")
    except FileNotFoundError:
        topik_dict['Semantic Signal Separation'] = pd.DataFrame()
    return topik_dict

# Load semua data
df_raw = load_raw_data()
df_stats = load_stats_data()
dict_topik = load_topic_data()

daftar_metrik = ['C_v', 'C_NPMI', 'Topic_Diversity', 'Topic_Quality']

# ==========================================
# SIDEBAR: NAVIGASI DASHBOARD
# ==========================================
st.sidebar.title("Menu")
st.sidebar.image("https://cdn-icons-png.flaticon.com/128/3579/3579187.png", width=50)
menu = st.sidebar.radio(
    "Choose a page:",
    ["Summary", "Detailed Information", "Computation Time", "Topic Exploration", "Raw Data"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 235 Undergraduate Thesis of PTIK 2020 - 2025")
st.sidebar.markdown("**Embedding:** IndoBERT")
st.sidebar.markdown("**Iteration:** 10 Random Seeds")

# ==========================================
# KONTEN HALAMAN UTAMA
# ==========================================

# HEADER DASHBOARD
st.title("📊 Topic Modeling Dashboard")
st.markdown("Comparison between **BERTopic**, **LDA**, and **Semantic Signal Separation**.")
st.markdown("---")

# 1. HALAMAN: RINGKASAN EKSEKUTIF
if menu == "Summary":
    st.header("🎯 Summary of Model Quality (Without Topic Interpretation)")
    
    # Menampilkan KPI Cards
    if not df_stats.empty:
        df_cv = df_stats[df_stats['Metric'] == 'C_v']
        df_td = df_stats[df_stats['Metric'] == 'Topic_Diversity']
        if not df_cv.empty:
            # Mencari model dengan C_v tertinggi
            best_cv_model = df_cv.loc[df_cv['Mean_Center'].idxmax()]
            best_td_model = df_td.loc[df_td['Mean_Center'].idxmax()]
            
            # Tampilan Kartu Metrik di atas
            kpi1, kpi2 = st.columns(2)
            kpi3, kpi4 = st.columns(2)
            kpi1.metric(label="🏆 Best Model by C_v", value=best_cv_model['Model'], delta=f"C_v: {best_cv_model['Mean_Center']:.3f}")
            kpi2.metric(label="🏆 Best Model by Topic Diversity", value=best_td_model['Model'], delta=f"Topic_Diversity: {best_td_model['Mean_Center']:.3f}")
            kpi3.metric(label="Total Document", value="235 Documents")
            kpi4.metric(label="Scenario for Evaluation", value="30 (3 Model x 10 Seed)")
            st.markdown("<br>", unsafe_allow_html=True)
            
        col1, col2 = st.columns(2)
        for i, metrik in enumerate(daftar_metrik):
            df_subset = df_stats[df_stats['Metric'] == metrik]
            if not df_subset.empty:
                fig_bar = px.bar(
                    df_subset, x="Model", y="Mean_Center", error_y="Std_Dev", color="Model",
                    title=f"Skor Agregat: {metrik}", text_auto='.3f',
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_bar.update_traces(
                    customdata=df_subset[["Minimum", "Maximum", "Mean_Lower", "Mean_Upper"]],
                    hovertemplate="<b>%{x}</b><br>Mean: %{y:.3f} ± %{error_y.value:.3f}<br>Min: %{customdata[0]:.3f}<br>Max: %{customdata[1]:.3f}<extra></extra>"
                )
                fig_bar.update_layout(uniformtext_minsize=10, uniformtext_mode='hide', margin=dict(t=50, b=20))
                
                if i % 2 == 0:
                    col1.plotly_chart(fig_bar, use_container_width=True)
                else:
                    col2.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Data statistik belum tersedia.")

# 2. HALAMAN: ANALISIS TREN KUALITAS
elif menu == "Detailed Information":
    st.header("📈 Based on Random Seed (Run 10-100)")
    
    if not df_raw.empty:
        col3, col4 = st.columns(2)
        for i, metrik in enumerate(daftar_metrik):
            if metrik in df_raw.columns:
                fig_line = px.line(
                    df_raw, x="Run", y=metrik, color="Model", markers=True, 
                    title=f"Stabilitas Skor {metrik}", color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_line.update_layout(margin=dict(t=50, b=20))
                if i % 2 == 0:
                    col3.plotly_chart(fig_line, use_container_width=True)
                else:
                    col4.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning(f"Kolom '{metrik}' tidak ditemukan.")
    else:
        st.info("Data mentah per iterasi tidak ditemukan.")

# 3. HALAMAN: WAKTU KOMPUTASI
elif menu == "Computation Time":
    st.header("⏱️ Information on Computation Time")
    st.markdown("*Model with lowest time and also consistent is the fastest*")
    
    if not df_raw.empty and 'Computation_Time' in df_raw.columns:
        fig_time = px.line(
            df_raw, x="Run", y="Computation_Time", color="Model", markers=True,
            title="Computation Time (iteration)", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_time.update_traces(hovertemplate="<b>%{x}</b><br>Run ke-%{x}<br>Waktu: %{y:.2f} detik<extra></extra>")
        st.plotly_chart(fig_time, use_container_width=True)
        
    if not df_stats.empty:
        df_waktu = df_stats[df_stats['Metric'] == 'Computation_Time']
        if not df_waktu.empty:
            fig_waktu = px.bar(
                df_waktu, x="Model", y="Mean_Center", error_y="Std_Dev", color="Model",
                title="Average time (second)", text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_waktu, use_container_width=True)

# 4. HALAMAN: EKSPLORASI TOPIK
elif menu == "Topic Exploration":
    st.header("📝 Topic Interpretation Exploration")
    st.markdown("Choose a model below to review the keywords extracted from the thesis document corpus.")
    
    pilihan_model = st.selectbox("Choose a model to be showed:", ["BERTopic", "LDA", "Semantic Signal Separation"])
    df_topik_pilihan = dict_topik.get(pilihan_model, pd.DataFrame())
    
    if not df_topik_pilihan.empty:
        st.dataframe(df_topik_pilihan, use_container_width=True, height=500)

        #Konversi dan download data topik
        csv_topics = df_topik_pilihan.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Topic results",
            data=csv_topics,
            file_name='topic_modeling_results.csv',
            mime='text/csv',
        )
    else:
        st.warning(f"Data topik untuk {pilihan_model} belum tersedia.")

# 5. HALAMAN: DATA MENTAH
elif menu == "Raw Data":
    st.header("📋 All data")
    st.markdown("Please download the data below for further analysis or reporting.")
    
    st.subheader("1. Data Statictics (Mean ± Std Dev)")
    if not df_stats.empty:
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        # Konversi DataFrame statistik menjadi format CSV yang siap diunduh
        csv_stats = df_stats.to_csv(index=False).encode('utf-8')
        
        # Menampilkan tombol unduh untuk data statistik
        st.download_button(
            label="📥 Download Statistical Data (CSV)",
            data=csv_stats,
            file_name='statistics_result_topic_modeling.csv',
            mime='text/csv',
        )
        
    st.markdown("---") # Garis pembatas visual
        
    st.subheader("2. Data Iteration (10 Random Seeds)")
    if not df_raw.empty:
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
        
        # Konversi DataFrame data mentah per iterasi menjadi format CSV
        csv_raw = df_raw.to_csv(index=False).encode('utf-8')
        
        # Menampilkan tombol unduh untuk data mentah
        st.download_button(
            label="📥 Download Iteration Data (CSV)",
            data=csv_raw,
            file_name='iteration_data_topic_modeling.csv',
            mime='text/csv',
        )