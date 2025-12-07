import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
from PIL import Image

# ============================================
# CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Bug Predictor",
    page_icon="🐛",
    layout="wide"
)

# Chemins
MODEL_DIR = r'C:\Users\M9-electro\Desktop\bug-predictor\models'
FIGURES_DIR = r'C:\Users\M9-electro\Desktop\bug-predictor\results\figures'
DATA_DIR = r'C:\Users\M9-electro\Desktop\bug-predictor\data\processed'

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'best_scaler.pkl'))
        
        with open(os.path.join(MODEL_DIR, 'best_model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata, True
    except Exception as e:
        return None, None, None, False

model, scaler, metadata, model_loaded = load_model()

# ============================================
# HEADER
# ============================================
st.title("🐛 Bug Predictor")
st.markdown("**Système Intelligent de Prédiction de Bugs**")
st.markdown("*Par : EZZAIM Saloua & ER-REMYTY Karima | Encadrante : Pr. MJAHED Soukaina*")
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Sélectionner une page",
    ["🏠 Accueil", "🔮 Prédiction", "📊 Performances"]
)

st.sidebar.markdown("---")

if model_loaded:
    st.sidebar.success("✅ Modèle chargé")
    st.sidebar.metric("Accuracy", f"{metadata['accuracy']:.2f}%")
    st.sidebar.metric("Recall", f"{metadata['recall']:.2f}%")
    st.sidebar.metric("F1-Score", f"{metadata['f1_score']:.2f}%")
else:
    st.sidebar.error("❌ Modèle non disponible")

# ============================================
# PAGE 1 : ACCUEIL
# ============================================
if page == "🏠 Accueil":
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    if model_loaded:
        col1.metric("🎯 Accuracy", f"{metadata['accuracy']:.2f}%", 
                    delta="✅ Objectif: ≥70%")
        col2.metric("📈 Recall", f"{metadata['recall']:.2f}%")
        col3.metric("🎲 F1-Score", f"{metadata['f1_score']:.2f}%")
        col4.metric("🤖 Modèle", metadata['model_name'].split('(')[0].strip())
    
    st.markdown("---")
    
    # Informations du projet
    st.subheader("📋 À Propos du Projet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎓 Contexte Académique**
        
        - **Université :** Cadi Ayyad
        - **Faculté :** Sciences Semlalia
        - **Formation :** Master IA
        - **Année :** 2024-2025
        
        **👥 Équipe**
        
        - EZZAIM Saloua
        - ER-REMYTY Karima
        
        **👨‍🏫 Encadrante**
        
        - Pr. MJAHED Soukaina
        """)
    
    with col2:
        st.success("""
        **🎯 Objectif**
        
        Prédire automatiquement les fichiers à risque dans un projet logiciel
        
        **📊 Dataset**
        
        - NASA Combined (13 projets)
        - 9,533 échantillons
        - 38 métriques de code
        
        **🤖 Modèle**
        
        - Random Forest Optimisé
        - class_weight='balanced'
        - Accuracy: 84.01% ✅
        """)
    
    st.markdown("---")
    
    # Méthodologie
    st.subheader("🛠️ Méthodologie & Organisation")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "SCRUM", "UML", "Architecture", "Design Patterns", "Structure Code"
    ])
    
    with tab1:
        st.markdown("""
        ### 📋 Méthodologie SCRUM
        
        **Product Backlog :**
        - Epic 1 : Extraction et Analyse des Données
        - Epic 2 : Modèle de Prédiction
        - Epic 3 : Interface et Visualisation
        - Epic 4 : API et Intégration
        
        **Sprints (3 x 2 semaines) :**
        
        **Sprint 1 - Fondations :**
        - Setup environnement
        - Extraction données Git
        - Calcul métriques de complexité
        - Pipeline de préparation
        
        **Sprint 2 - Modèle ML :**
        - Feature engineering
        - Entraînement Random Forest
        - Comparaison algorithmes
        - Validation croisée
        
        **Sprint 3 - Interface :**
        - Dashboard Streamlit
        - Visualisations
        - Documentation
        - Tests finaux
        
        **Ceremonies SCRUM :**
        - Daily Stand-up (15 min/jour)
        - Sprint Planning (4h/sprint)
        - Sprint Review (2h/sprint)
        - Sprint Retrospective (1h30/sprint)
        """)
    
    with tab2:
        st.markdown("""
        ### 📐 Conception UML
        
        **Diagrammes Réalisés :**
        
        **1. Diagramme de Cas d'Utilisation**
        - Acteurs : Développeur, Chef de Projet, Système CI/CD
        - Cas d'usage : Analyser Projet, Visualiser Risques, Générer Rapport
        
        **2. Diagramme de Classes**
        - DataExtractor (Abstract)
        - GitExtractor, SVNExtractor
        - FileAnalyzer
        - BugPredictor
        - PredictionStrategy (Interface)
        - Project, FileInfo, Report
        
        **3. Diagramme de Séquence**
        - Séquence : Analyse d'un projet
        - Séquence : Entraînement du modèle
        
        **4. Diagramme d'Activité**
        - Processus de prédiction complet
        
        **5. Diagramme de Composants**
        - Architecture en couches (Présentation, Métier, Données)
        """)
    
    with tab3:
        st.markdown("""
        ### 🏗️ Architecture
        
        **Architecture Logique (MVC adapté) :**
```
        ┌─────────────────────────────────┐
        │   Couche Présentation           │
        │   (Streamlit Dashboard)         │
        ├─────────────────────────────────┤
        │   Couche Métier                 │
        │   (BugPredictor, Services)      │
        ├─────────────────────────────────┤
        │   Couche Données                │
        │   (DataExtractor, CSV, Models)  │
        └─────────────────────────────────┘
```
        
        **Architecture Physique :**
        
        - **Application Streamlit :** Port 8501
        - **Modèles ML :** Fichiers .pkl locaux
        - **Base de données :** CSV (data.csv)
        - **Figures :** PNG (visualisations)
        
        **Technologies :**
        
        - **Backend :** Python 3.10+
        - **ML :** Scikit-learn, imbalanced-learn
        - **Frontend :** Streamlit
        - **Data :** Pandas, NumPy
        - **Viz :** Matplotlib, Seaborn
        """)
    
    with tab4:
        st.markdown("""
        ### 🎨 Design Patterns
        
        **Patterns Utilisés :**
        
        **1. Strategy Pattern**
        - Interface : `PredictionStrategy`
        - Implémentations : 
          - `RandomForestStrategy`
          - `SVMStrategy`
          - `GradientBoostingStrategy`
        - Permet de changer dynamiquement l'algorithme
        
        **2. Factory Pattern**
        - `DataExtractorFactory`
        - Crée le bon extracteur (Git, SVN) selon le type
        
        **3. Singleton Pattern**
        - Classe `Config`
        - Une seule instance de configuration globale
        
        **4. Observer Pattern**
        - Notifications de progrès d'analyse
        - Mise à jour de l'interface en temps réel
        
        **Justification :**
        - **Flexibilité :** Changer de modèle facilement
        - **Maintenabilité :** Code modulaire
        - **Extensibilité :** Ajouter de nouveaux algorithmes
        """)
    

    st.markdown("---")
    
    # Visualisations
    st.subheader("📈 Visualisations")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribution du Dataset**")
            img1 = Image.open(os.path.join(FIGURES_DIR, '01_dataset_distribution.png'))
            st.image(img1, use_container_width=True)
        
        with col2:
            st.markdown("**Performances du Modèle**")
            img2 = Image.open(os.path.join(FIGURES_DIR, '02_model_performance.png'))
            st.image(img2, use_container_width=True)
        
        st.markdown("**Comparaison des Modèles**")
        img3 = Image.open(os.path.join(FIGURES_DIR, '03_model_comparison.png'))
        st.image(img3, use_container_width=True)
        
    except:
        st.warning("⚠️ Visualisations non disponibles. Exécutez generate_visualizations.py")

# ============================================
# PAGE 2 : PRÉDICTION
# ============================================
elif page == "🔮 Prédiction":
    st.header("🔮 Prédiction de Bugs")
    
    if not model_loaded:
        st.error("❌ Modèle non chargé. Impossible de faire des prédictions.")
        st.stop()
    
    # Charger les features
    df = pd.read_csv(os.path.join(DATA_DIR, 'data.csv'))
    df = df.dropna(subset=['Defective'])
    X = df.drop(columns=['Defective', 'source', 'label'], errors='ignore')
    feature_names = X.columns.tolist()
    
    st.info(f"📋 Le modèle utilise **{len(feature_names)} métriques** de code source.")
    
    # Onglets
    tab1, tab2 = st.tabs(["📤 Upload CSV", "✍️ Saisie Manuelle"])
    
    # TAB 1 : Upload CSV
    with tab1:
        st.subheader("📤 Uploader un fichier CSV")
        
        st.markdown("""
        **Format attendu :** Un fichier CSV contenant les 38 métriques de code.
        
        Les colonnes doivent correspondre aux features du modèle.
        """)
        
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé : **{len(input_df)} fichiers**")
                
                # Vérifier colonnes
                missing_cols = set(feature_names) - set(input_df.columns)
                
                if missing_cols:
                    st.error(f"❌ Colonnes manquantes : {missing_cols}")
                else:
                    # Bouton prédiction
                    if st.button("🔮 Prédire les Bugs", type="primary"):
                        with st.spinner("Prédiction en cours..."):
                            # Préparation
                            input_scaled = scaler.transform(input_df[feature_names])
                            
                            # Prédictions
                            predictions = model.predict(input_scaled)
                            probabilities = model.predict_proba(input_scaled)[:, 1]
                            
                            # Résultats
                            results_df = pd.DataFrame({
                                'Fichier': range(1, len(predictions)+1),
                                'Prédiction': ['🐛 BUG' if p == 1 else '✅ OK' for p in predictions],
                                'Probabilité Bug (%)': (probabilities * 100).round(2),
                                'Niveau de Risque': [
                                    '🔴 ÉLEVÉ' if prob > 0.7 
                                    else ('🟡 MOYEN' if prob > 0.4 else '🟢 FAIBLE')
                                    for prob in probabilities
                                ]
                            })
                            
                            st.markdown("---")
                            st.subheader("📊 Résultats de la Prédiction")
                            
                            # Métriques globales
                            col1, col2, col3 = st.columns(3)
                            
                            n_bugs = sum(predictions)
                            n_total = len(predictions)
                            n_high_risk = sum(probabilities > 0.7)
                            
                            col1.metric("Bugs Détectés", f"{n_bugs} / {n_total}")
                            col2.metric("Pourcentage", f"{n_bugs/n_total*100:.1f}%")
                            col3.metric("Risque Élevé", n_high_risk)
                            
                            # Table des résultats
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Téléchargement
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Télécharger les résultats (CSV)",
                                data=csv,
                                file_name="bug_predictions.csv",
                                mime="text/csv"
                            )
                
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
    
    # TAB 2 : Saisie manuelle
    with tab2:
        st.subheader("✍️ Saisie Manuelle des Métriques (Temps Réel)")
        st.warning("⚠️ Les résultats se mettent à jour dès que vous modifiez une valeur.")

        col1, col2 = st.columns(2)
        inputs = {}

        main_features = [
            'LOC_TOTAL', 'CYCLOMATIC_COMPLEXITY', 'LOC_EXECUTABLE',
            'HALSTEAD_VOLUME', 'HALSTEAD_DIFFICULTY', 'NUMBER_OF_LINES',
            'LOC_COMMENTS', 'BRANCH_COUNT'
        ]

        for i, feature in enumerate(feature_names):
            if feature in main_features:
                if i % 2 == 0:
                    inputs[feature] = col1.number_input(
                        f"📊 {feature}", value=0.0, format="%.2f"
                    )
                else:
                    inputs[feature] = col2.number_input(
                        f"📊 {feature}", value=0.0, format="%.2f"
                    )
            else:
                inputs[feature] = 0.0  # Valeurs par défaut

        # Prédiction automatique
        try:
            input_df = pd.DataFrame([inputs])
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0, 1]

            st.markdown("---")
            st.subheader("📊 Résultat")

            if prediction == 1:
                st.error(f"🐛 BUG détecté ({probability * 100:.2f}% de probabilité)")
            else:
                st.success(f"✅ Pas de bug ({probability * 100:.2f}% de probabilité)")

        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# ============================================
# PAGE 3 : PERFORMANCES
# ============================================
else:  # page == "📊 Performances"
    st.header("📊 Performances du Modèle")
    
    # Comparaison des modèles
    try:
        comparison_df = pd.read_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'))
        
        st.subheader("🔬 Comparaison des Algorithmes")
        
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
            use_container_width=True
        )
        
        # Graphique de comparaison
        st.markdown("---")
        st.subheader("📈 Graphique Comparatif")
        
        import plotly.express as px
        
        fig = px.bar(
            comparison_df,
            x='Modèle',
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            barmode='group',
            title='Comparaison des Métriques par Modèle',
            labels={'value': 'Score (%)', 'variable': 'Métrique'},
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Meilleur modèle
        st.markdown("---")
        best_row = comparison_df.iloc[0]
        
        st.success(f"""
        ### 🏆 Meilleur Modèle : {best_row['Modèle']}
        
        - **Accuracy :** {best_row['Accuracy']:.2f}%
        - **Precision :** {best_row['Precision']:.2f}%
        - **Recall :** {best_row['Recall']:.2f}%
        - **F1-Score :** {best_row['F1-Score']:.2f}%
        - **Temps d'entraînement :** {best_row['Temps (s)']:.2f}s
        """)
        
    except:
        st.error("❌ Fichier de comparaison non trouvé")
    
    st.markdown("---")
    
    # Métriques Train vs Test
    st.subheader("🏋️ Train vs Test")
    
    try:
        metrics_df = pd.read_csv(os.path.join(MODEL_DIR, 'metrics_comparison.csv'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📚 Entraînement")
            for _, row in metrics_df.iterrows():
                st.metric(row['Metric'], f"{row['Train']*100:.2f}%")
        
        with col2:
            st.markdown("### 🧪 Test")
            for _, row in metrics_df.iterrows():
                gap = (row['Train'] - row['Test']) * 100
                st.metric(
                    row['Metric'], 
                    f"{row['Test']*100:.2f}%",
                    delta=f"{gap:.2f}% gap",
                    delta_color="inverse"
                )
        
        # Diagnostic Overfitting
        st.markdown("---")
        st.subheader("🔍 Diagnostic Overfitting")
        
        acc_gap = (metrics_df[metrics_df['Metric'] == 'Accuracy']['Train'].values[0] - 
                   metrics_df[metrics_df['Metric'] == 'Accuracy']['Test'].values[0]) * 100
        
        if acc_gap > 10:
            st.warning(f"⚠️ **Overfitting détecté** : Écart de {acc_gap:.2f}% entre Train et Test")
        elif acc_gap > 5:
            st.info(f"ℹ️ **Overfitting léger** : Écart de {acc_gap:.2f}%")
        else:
            st.success(f"✅ **Pas d'overfitting** : Écart de {acc_gap:.2f}%")
    
    except:
        st.warning("⚠️ Métriques de comparaison non disponibles")
    
    st.markdown("---")
    
    # Atteinte des objectifs
    st.subheader("🎯 Atteinte des Objectifs")
    
    if model_loaded:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if metadata['accuracy'] >= 70:
                st.success(f"""
                **✅ OBJECTIF ATTEINT**
                
                Accuracy : {metadata['accuracy']:.2f}% ≥ 70%
                """)
            else:
                st.error(f"""
                **❌ Objectif non atteint**
                
                Accuracy : {metadata['accuracy']:.2f}% < 70%
                """)
        
        with col2:
            if metadata['recall'] >= 50:
                st.success(f"""
                **✅ RECALL EXCELLENT**
                
                {metadata['recall']:.2f}% ≥ 50%
                """)
            elif metadata['recall'] >= 40:
                st.info(f"""
                **ℹ️ RECALL ACCEPTABLE**
                
                {metadata['recall']:.2f}% (40-50%)
                """)
            else:
                st.warning(f"""
                **⚠️ RECALL FAIBLE**
                
                {metadata['recall']:.2f}% < 40%
                """)
        
        with col3:
            st.info(f"""
            **📊 F1-SCORE**
            
            {metadata['f1_score']:.2f}%
            
            (Équilibre Precision/Recall)
            """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>© 2024-2025 Bug Predictor | Université Cadi Ayyad - FSS Marrakech</p>
    <p>Développé par EZZAIM Saloua & ER-REMYTY Karima</p>
</div>
""", unsafe_allow_html=True)