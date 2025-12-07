"""
Fixtures pytest partagées pour tous les tests.

Les fixtures sont des fonctions qui fournissent des données ou objets
réutilisables pour les tests.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tempfile
import os


# ============================================
# FIXTURES DE DONNÉES
# ============================================

@pytest.fixture
def sample_data():
    """
    Crée un petit dataset de test avec 100 échantillons.

    Returns:
        pd.DataFrame: Dataset avec features et target
    """
    np.random.seed(42)

    # 100 échantillons, 5 features
    data = {
        'LOC_TOTAL': np.random.randint(10, 500, 100),
        'CYCLOMATIC_COMPLEXITY': np.random.randint(1, 20, 100),
        'HALSTEAD_VOLUME': np.random.uniform(10, 1000, 100),
        'HALSTEAD_DIFFICULTY': np.random.uniform(1, 50, 100),
        'LOC_EXECUTABLE': np.random.randint(5, 400, 100),
        'Defective': np.random.choice([0, 1], 100, p=[0.85, 0.15])
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_nan():
    """Dataset avec valeurs manquantes pour tester le nettoyage."""
    np.random.seed(42)

    data = {
        'LOC_TOTAL': [100, 200, np.nan, 150, 300],
        'CYCLOMATIC_COMPLEXITY': [5, np.nan, 8, 12, 15],
        'HALSTEAD_VOLUME': [50.5, 100.2, 75.3, np.nan, 200.1],
        'Defective': [0, 1, 0, np.nan, 1]
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_duplicates():
    """Dataset avec doublons pour tester la suppression."""
    data = {
        'LOC_TOTAL': [100, 200, 100, 150],  # Ligne 1 et 3 identiques
        'CYCLOMATIC_COMPLEXITY': [5, 10, 5, 8],
        'Defective': [0, 1, 0, 1]
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path, sample_data):
    """
    Crée un fichier CSV temporaire pour les tests.

    Args:
        tmp_path: Fixture pytest pour dossier temporaire
        sample_data: Fixture de données

    Returns:
        str: Chemin du fichier CSV
    """
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def balanced_data():
    """Dataset équilibré (50/50) pour tests spécifiques."""
    np.random.seed(42)

    n_samples = 100
    n_positive = 50

    data = {
        'LOC_TOTAL': np.random.randint(10, 500, n_samples),
        'CYCLOMATIC_COMPLEXITY': np.random.randint(1, 20, n_samples),
        'Defective': [1] * n_positive + [0] * (n_samples - n_positive)
    }

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # Mélanger


# ============================================
# FIXTURES DE MODÈLES ML
# ============================================

@pytest.fixture
def trained_model(sample_data):
    """
    Modèle Random Forest pré-entraîné pour tests.

    Returns:
        RandomForestClassifier: Modèle entraîné
    """
    X = sample_data.drop('Defective', axis=1)
    y = sample_data['Defective']

    model = RandomForestClassifier(
        n_estimators=10,  # Peu d'arbres pour tests rapides
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)
    return model


@pytest.fixture
def scaler_fitted(sample_data):
    """
    Scaler pré-entraîné pour tests.

    Returns:
        StandardScaler: Scaler fitted
    """
    X = sample_data.drop('Defective', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def model_file(tmp_path, trained_model):
    """
    Sauvegarde un modèle dans un fichier temporaire.

    Returns:
        str: Chemin du fichier .pkl
    """
    import joblib
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(trained_model, model_path)
    return str(model_path)


# ============================================
# FIXTURES DE CONFIGURATION
# ============================================

@pytest.fixture
def temp_config_dir(tmp_path):
    """
    Crée une structure de dossiers temporaire pour tests.

    Returns:
        dict: Chemins des dossiers
    """
    structure = {
        'data_raw': tmp_path / "data" / "raw",
        'data_processed': tmp_path / "data" / "processed",
        'models': tmp_path / "models",
        'results': tmp_path / "results" / "figures"
    }

    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)

    return structure


@pytest.fixture
def mock_config(temp_config_dir):
    """
    Configuration mockée pour tests.

    Returns:
        dict: Configuration de test
    """
    return {
        'MODEL_PATH': str(temp_config_dir['models'] / 'test_model.pkl'),
        'SCALER_PATH': str(temp_config_dir['models'] / 'test_scaler.pkl'),
        'DATA_PATH': str(temp_config_dir['data_processed'] / 'test_data.csv'),
        'TEST_SIZE': 0.2,
        'RANDOM_STATE': 42,
        'THRESHOLD': 0.5
    }


# ============================================
# FIXTURES UTILITAIRES
# ============================================

@pytest.fixture
def suppress_warnings():
    """Supprime les warnings pendant les tests."""
    import warnings
    warnings.filterwarnings('ignore')
    yield
    warnings.filterwarnings('default')


@pytest.fixture(autouse=True)
def reset_random_state():
    """Réinitialise le seed aléatoire avant chaque test."""
    np.random.seed(42)
    import random
    random.seed(42)


# ============================================
# MARKERS PERSONNALISÉS
# ============================================

def pytest_configure(config):
    """Configure les markers personnalisés."""
    config.addinivalue_line(
        "markers", "unit: Tests unitaires simples et rapides"
    )
    config.addinivalue_line(
        "markers", "integration: Tests d'intégration (plus lents)"
    )
    config.addinivalue_line(
        "markers", "slow: Tests lents (>1 seconde)"
    )