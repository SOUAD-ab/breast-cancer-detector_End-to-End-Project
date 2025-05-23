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

# Appliquer le fond (assure-toi que ce chemin est correct)
set_bg_image("images/illustration_fond.jpg")

# Afficher le logo (assure-toi que ce chemin est correct)
st.image("images/logo_um5.png", width=150)

# 📁 Chargement du modèle fine-tuné (cache pour optimiser les rechargements)
MODEL_PATH = os.path.join("notebook_dl.ipynb", "fine_tuned_model.keras")
class_names = {0: "Bénin", 1: "Malin", 2: "Normal"}

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        return None

# 🧪 Prétraitement de l’image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# 🧾 Affiche les recommandations selon la classe prédite
def display_recommendations(status):
    if status == "Malin":
        st.error("**Recommandation :** Consultation oncologique urgente requise.")
    elif status == "Bénin":
        st.warning("**Recommandation :** Surveillance régulière recommandée.")
    else:  # Normal
        st.success("**Recommandation :** Aucune action immédiate nécessaire.")

# 🧬 Fonction principale de l’app
def main():
    st.title("Détection de Cancer du Sein par Intelligence Artificielle")
    st.caption("Modèle : MobileNetV2 fine-tuné sur le dataset BUSI")

    st.markdown("**Outil d'aide au diagnostic**\nChargez une image échographique pour obtenir une analyse automatisée.")

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        model = load_model()
        if model is None:
            st.error("Le modèle n’a pas pu être chargé, veuillez vérifier le fichier.")
            return
        
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Le fichier uploadé n'est pas une image valide : {e}")
            return

        col1, col2 = st.columns([1, 1.5], gap="large")
        with col1:
            st.image(image, use_container_width=True, caption="Image analysée")

        with col2:
            try:
                processed_image = preprocess_image(image)
                
                with st.spinner("Analyse en cours..."):
                    prediction = model.predict(processed_image)
                
                pred_class = np.argmax(prediction)
                confidence = float(np.max(prediction)) * 100
                status = class_names[pred_class]

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                status_config = {
                    "Bénin": {"color": "#28a745", "icon": "🟢"},
                    "Malin": {"color": "#dc3545", "icon": "🔴"},
                    "Normal": {"color": "#17a2b8", "icon": "🔵"}
                }

                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                        <span style="font-size: 3rem;">{status_config[status]['icon']}</span>
                        <div>
                            <h2 style="color: {status_config[status]['color']}; margin: 0;">{status.upper()}</h2>
                            <p style="margin-top: 0.3rem; color: #6c757d;">Confiance : {confidence:.1f}%</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="confidence-bar" style="width: {confidence:.0f}%;"></div>', unsafe_allow_html=True)

                display_recommendations(status)

                st.markdown('</div>', unsafe_allow_html=True)

                # 📊 Graphique des probabilités
                st.markdown("### Visualisation des probabilités")
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#28a745', '#dc3545', '#17a2b8']
                labels = ["🟢 Bénin", "🔴 Malin", "🔵 Normal"]
                bars = ax.bar(labels, prediction[0] * 100, color=colors)
                ax.set_ylabel('Probabilité (%)', fontweight='bold')
                ax.set_ylim(0, 100)
                plt.xticks(fontweight='bold')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}%', ha='center', va='bottom')
                st.pyplot(fig)
                plt.close(fig)

                # 🧾 Détail texte brut
                st.markdown("### Détails des prédictions :")
                for i, p in enumerate(prediction[0]):
                    st.write(f"→ {class_names[i]} : {p * 100:.2f}%")

            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {str(e)}")

    # 📚 Infos complémentaires
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Classes détectées**")
        st.markdown("- 🔴 Malin\n- 🟢 Bénin\n- 🔵 Normal")
    with col2:
        st.markdown("**Performances du modèle**")
        st.markdown("- Précision (test) : 86.5%\n- MobileNetV2 fine-tuné\n- Dataset BUSI")
    with col3:
        st.markdown("**Avertissement médical**")
        st.markdown("Cet outil ne remplace pas un diagnostic médical professionnel.\nConsultez toujours un spécialiste qualifié.")

if __name__ == "__main__":
    main()
