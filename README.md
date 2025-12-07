# 🐛 Bug Predictor

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Système Intelligent de Prédiction Automatique des Fichiers à Risque dans les Projets Logiciels**

Projet académique réalisé dans le cadre du Master Intelligence Artificielle - Faculté des Sciences Semlalia, Université Cadi Ayyad, Marrakech.

---

## 📖 Table des Matières

- [À Propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Résultats](#résultats)
- [Démo](#démo)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Méthodologie](#méthodologie)
- [Technologies](#technologies)
- [Contributeurs](#contributeurs)
- [Licence](#licence)

---

## 🎯 À Propos

Bug Predictor est un système de Machine Learning capable de prédire automatiquement quels fichiers d'un projet logiciel sont susceptibles de contenir des bugs. Le système analyse des métriques de code statiques (complexité cyclomatique, métriques de Halstead, LOC) pour identifier les zones à risque et aider les équipes de développement à prioriser leurs efforts de test et de revue de code.

### 🎓 Contexte Académique

- **Université** : Cadi Ayyad - Faculté des Sciences Semlalia, Marrakech
- **Formation** : Master Spécialisé Intelligence Artificielle
- **Année** : 2024-2025
- **Encadrante** : Pr. MJAHED Soukaina
- **Étudiantes** : 
  - EZZAIM Saloua
  - ER-REMYTY Karima

### 🎯 Objectifs

- ✅ Prédire les fichiers à risque avec une **accuracy ≥ 70%**
- ✅ Fournir une interface intuitive pour les développeurs
- ✅ Intégrer des techniques avancées de ML (gestion du déséquilibre, optimisation)
- ✅ Appliquer une méthodologie SCRUM rigoureuse
- ✅ Respecter les principes de génie logiciel (UML, Design Patterns, architecture en couches)

---

## ✨ Fonctionnalités

### 🔮 Modes de Prédiction

1. **Upload CSV** : Analyse batch de plusieurs fichiers simultanément
2. **Analyse Git** : Extraction automatique depuis un repository (structure préparée)
3. **Saisie Manuelle** : Prédiction instantanée avec métriques personnalisées

### 📊 Visualisations

- Distribution du dataset (bugs vs no bugs)
- Performances du modèle (métriques, matrice de confusion)
- Comparaison de 4 algorithmes ML
- Importance des features

### 🎨 Interface Utilisateur

- Dashboard web interactif (Streamlit)
- 3 pages : Accueil, Prédiction, Performances
- Export des résultats en CSV
- Graphiques interactifs (Plotly)

---

## 🏆 Résultats

### Performances du Modèle

| Métrique | Train | Test | Objectif |
|----------|-------|------|----------|
| **Accuracy** | 94.85% | **84.01%** | ✅ ≥70% |
| **Precision** | 75.31% | 44.44% | - |
| **Recall** | 95.24% | 46.15% | - |
| **F1-Score** | 84.11% | 45.24% | - |

### Dataset

- **Source** : 13 projets NASA combinés (PC1-5, KC1-4, CM1, MC1-2, JM1, MW1)
- **Échantillons** : 9,533 (après nettoyage)
- **Features** : 38 métriques de code
- **Distribution** : 14.32% bugs, 85.68% no bugs

### Comparaison avec la Littérature

Notre modèle atteint des performances **comparables ou supérieures** aux publications scientifiques sur les mêmes datasets NASA :

| Étude | Accuracy | Recall | F1-Score |
|-------|----------|--------|----------|
| Menzies et al. (2007) | 70-80% | 40-70% | 42-70% |
| Zimmermann et al. (2007) | 75-85% | 35-65% | 38-68% |
| D'Ambros et al. (2012) | 78-82% | 42-58% | 40-62% |
| **Notre projet** | **84.01%** | **46.15%** | **45.24%** |

---

## 🎥 Démo

### Captures d'Écran

#### Page d'Accueil
![Accueil](docs/screenshots/home.png)

#### Prédiction CSV
![Prediction CSV](docs/screenshots/prediction_csv.png)

#### Saisie Manuelle
![Saisie Manuelle](docs/screenshots/manual_input.png)

#### Performances
![Performances](docs/screenshots/performance.png)

---

## 🚀 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Git

### Installation Rapide

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/bug-predictor.git
cd bug-predictor

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Vérifier l'installation
python -c "import sklearn, pandas, streamlit; print('✅ Installation réussie')"
```

### Télécharger les Datasets NASA (Optionnel)

Si vous souhaitez ré-entraîner le modèle :

```bash
# Les datasets ARFF doivent être placés dans data/raw/
# Téléchargement : https://zenodo.org/record/268460
```

---

## 💻 Utilisation

### Lancer l'Application

```bash
streamlit run app_simple.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

### Pipeline Complet (Ré-entraînement)

Si vous avez les datasets NASA et souhaitez ré-entraîner :

```bash
# 1. Conversion et fusion des datasets
python scripts/convert_and_merge_arff.py

# 2. Entraînement du modèle
python scripts/train_model.py

# 3. Comparaison des algorithmes
python scripts/model_comparison.py

# 4. Génération des visualisations
python scripts/generate_visualizations.py

# 5. Lancer l'application
streamlit run app_simple.py
```

### Utilisation de l'Interface

#### Mode 1 : Upload CSV

1. Accédez à la page **"🔮 Prédiction"**
2. Cliquez sur l'onglet **"📤 Upload CSV"**
3. Sélectionnez votre fichier CSV contenant les métriques
4. Cliquez sur **"🔮 Prédire les Bugs"**
5. Consultez les résultats et téléchargez-les en CSV

#### Mode 2 : Saisie Manuelle

1. Accédez à l'onglet **"✍️ Saisie Manuelle"**
2. Remplissez les champs avec les métriques de votre fichier
3. Cliquez sur **"🔮 Prédire"**
4. Consultez le résultat : prédiction + probabilité + niveau de risque

---

## 🏗️ Architecture

### Structure du Projet

```
bug-predictor/
├── data/
│   ├── raw/              # Datasets ARFF bruts
│   └── processed/        # Données traitées (CSV)
├── models/               # Modèles ML entraînés (.pkl)
├── results/
│   └── figures/          # Graphiques générés
├── scripts/              # Scripts d'exécution
│   ├── convert_and_merge_arff.py
│   ├── train_model.py
│   ├── model_comparison.py
│   └── generate_visualizations.py
├── src/                  # Code source modulaire
│   ├── data/             # Gestion des données
│   ├── models/           # Modèles ML
│   ├── ui/               # Interface utilisateur
│   └── utils/            # Utilitaires
├── tests/                # Tests unitaires
├── docs/                 # Documentation
│   ├── uml/              # Diagrammes UML
│   ├── scrum/            # Artefacts SCRUM
│   └── rapport/          # Rapport LaTeX
├── app_simple.py         # Application Streamlit
├── requirements.txt      # Dépendances Python
└── README.md            # Ce fichier
```

### Architecture en Couches

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

### Design Patterns Appliqués

1. **Strategy Pattern** : Algorithmes ML interchangeables
2. **Factory Pattern** : Création d'extracteurs de données
3. **Singleton Pattern** : Configuration globale
4. **Template Method Pattern** : Processus d'entraînement standardisé

---

## 📐 Méthodologie

### SCRUM

Le projet a été développé en suivant rigoureusement la méthodologie SCRUM :

- **3 sprints de 2 semaines** chacun
- **Product Backlog** : 18 User Stories réparties en 4 Epics
- **Cérémonies complètes** : Daily Stand-ups, Sprint Planning, Sprint Review, Sprint Retrospective
- **Artefacts** : Burndown Charts, Velocity tracking, Definition of Done

### Conception UML

- Diagramme de cas d'utilisation
- Diagramme de classes
- Diagrammes de séquence (2)
- Diagramme d'activité
- Diagramme de composants

### Principes SOLID

- **S**ingle Responsibility
- **O**pen/Closed
- **L**iskov Substitution
- **I**nterface Segregation
- **D**ependency Inversion

---

## 🛠️ Technologies

### Machine Learning

- **Scikit-learn 1.3.2** : Algorithmes ML, prétraitement, évaluation
- **Imbalanced-learn 0.11.0** : Gestion du déséquilibre des classes
- **Joblib 1.3.2** : Sérialisation des modèles

### Data Science

- **Pandas 2.1.3** : Manipulation de données
- **NumPy 1.26.2** : Calculs numériques
- **SciPy 1.11.4** : Conversion ARFF

### Visualisation

- **Matplotlib 3.8.2** : Graphiques statiques
- **Seaborn 0.13.0** : Visualisations statistiques
- **Plotly 5.18.0** : Graphiques interactifs

### Interface

- **Streamlit 1.29.0** : Application web interactive

### Développement

- **Python 3.10+** : Langage principal
- **Git** : Contrôle de version
- **pytest** : Tests unitaires
- **LaTeX** : Documentation académique

---

## 👥 Contributeurs

| Nom | Rôle | Email |
|-----|------|-------|
| **EZZAIM Saloua** | Scrum Master & Developer | ezzaimsaloua@... |
| **ER-REMYTY Karima** | Developer | erremytykarima@gmail.com |

### Encadrante

- **Pr. MJAHED Soukaina** - Faculté des Sciences Semlalia

---

## 📚 Documentation

### Rapport Complet

Le rapport académique complet (80+ pages) est disponible dans le dossier `docs/rapport/`.

### Diagrammes UML

Tous les diagrammes UML sont disponibles dans `docs/uml/`.

### Artefacts SCRUM

Product Backlog, Sprint Backlogs, et Retrospectives dans `docs/scrum/`.

---

## 🔮 Perspectives Futures

### Court Terme

- [ ] Amélioration du Recall (objectif : 60%)
- [ ] Extraction Git complète
- [ ] Tests unitaires complets (80% couverture)
- [ ] Documentation utilisateur détaillée

### Moyen Terme

- [ ] API REST avec FastAPI
- [ ] Intégration CI/CD (GitHub Actions, Jenkins)
- [ ] Support multi-langages (Java, Python, JavaScript)
- [ ] Enrichissement des features (métriques Git)

### Long Terme

- [ ] Deep Learning (CNN, Transformers)
- [ ] Analyse sémantique du code (AST)
- [ ] Système de recommandations
- [ ] Apprentissage continu
- [ ] Déploiement cloud (AWS, Azure)
- [ ] Plateforme SaaS

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Pr. MJAHED Soukaina** pour son encadrement rigoureux
- **Faculté des Sciences Semlalia** pour la formation de qualité
- **NASA** pour les datasets publics
- **Communauté open-source** pour les outils et bibliothèques

---

## 📞 Contact

Pour toute question ou collaboration :

- 📧 Email : ezzaimsaloua@... | erremytykarima@gmail.com
- 🎓 Université : Cadi Ayyad - FSS Marrakech
- 📅 Année : 2024-2025

---

## 📊 Statistiques du Projet

![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-2150-blue)
![Files](https://img.shields.io/badge/Files-47-green)
![Commits](https://img.shields.io/badge/Commits-50+-orange)

---

<div align="center">

**⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile ! ⭐**

Développé avec ❤️ par EZZAIM Saloua & ER-REMYTY Karima

</div>