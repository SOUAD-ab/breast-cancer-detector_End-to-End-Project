import os
import base64
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 🎨 Configuration de la page
st.set_page_config(
    page_title="Détection de Cancer du Sein",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# 🔄 Fonction pour encoder une image en Base64 (pour le fond)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 🎨 Appliquer un fond personnalisé
def set_bg_image(png_path):
    bin_str = get_base64_of_bin_file(png_path)
    st.markdown(f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main-container {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 2.5rem 3rem;
            margin: 2rem auto;
            max-width: 1100px;
        }}
        .upload-section {{
            background: #f1f8ff;
            border: 2px dashed #2A5C82;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
            transition: border-color 0.3s ease;
        }}
        .upload-section:hover {{
            border-color: #1b3d5b;
        }}
        .result-card {{
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            margin-top: 1.5rem;
        }}
        .confidence-bar {{
            height: 25px;
            border-radius: 12px;
            background: linear-gradient(90deg, #e0f3f8 0%, #2A5C82 100%);
            margin-top: 0.4rem;
        }}
        </style>
    ''', unsafe_allow_html=True)

# Appliquer le fond (vérifie que le chemin est correct)
set_bg_image("images/illustration_fond.jpg")

# Afficher le logo (vérifie que le chemin est correct)
st.image("images/logo_um5.png", width=150)

# 📁 Chargement du modèle fine-tuné (cache pour optimiser les recharg
