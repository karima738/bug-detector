"""
Tests unitaires pour le module DataProcessor.

Teste toutes les fonctionnalités de prétraitement des données.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Simuler la classe DataProcessor (adaptez selon votre implémentation)
class DataProcessor:
    """Classe simplifiée pour les tests."""

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """Charge les données depuis un CSV."""
        if self.filepath:
            self.data = pd.read_csv(self.filepath)
        return self.data

    def clean_data(self, df):
        """Nettoie les données."""
        # Supprimer NaN dans Defective
        df = df.dropna(subset=['Defective'])

        # Supprimer doublons
        df = df.drop_duplicates()

        # Remplir NaN avec médiane
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Defective':
                df[col] = df[col].fillna(df[col].median())

        return df

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Sépare en train/test."""
        return train_test_split(X, y, test_size=test_size,
                                stratify=y, random_state=random_state)


# ============================================
# TESTS DE CHARGEMENT
# ============================================

@pytest.mark.unit
class TestDataLoading:
    """Tests du chargement des données."""

    def test_load_data_from_csv(self, sample_csv_file):
        """Test que load_data charge correctement un CSV."""
        processor = DataProcessor(sample_csv_file)
        df = processor.load_data()

        assert df is not None, "Le DataFrame ne devrait pas être None"
        assert isinstance(df, pd.DataFrame), "Devrait retourner un DataFrame"
        assert len(df) == 100, "Devrait avoir 100 lignes"
        assert 'Defective' in df.columns, "Devrait contenir la colonne Defective"

    def test_load_data_invalid_path(self):
        """Test avec un chemin invalide."""
        processor = DataProcessor("chemin/inexistant.csv")

        with pytest.raises(FileNotFoundError):
            processor.load_data()

    def test_load_data_columns_present(self, sample_csv_file):
        """Test que toutes les colonnes attendues sont présentes."""
        processor = DataProcessor(sample_csv_file)
        df = processor.load_data()

        expected_cols = ['LOC_TOTAL', 'CYCLOMATIC_COMPLEXITY',
                         'HALSTEAD_VOLUME', 'Defective']

        for col in expected_cols:
            assert col in df.columns, f"Colonne {col} manquante"


# ============================================
# TESTS DE NETTOYAGE
# ============================================

@pytest.mark.unit
class TestDataCleaning:
    """Tests du nettoyage des données."""

    def test_remove_nan_in_target(self, sample_data_with_nan):
        """Test suppression des NaN dans Defective."""
        processor = DataProcessor()
        df_clean = processor.clean_data(sample_data_with_nan)

        assert df_clean['Defective'].isnull().sum() == 0, \
            "Ne devrait pas avoir de NaN dans Defective"
        assert len(df_clean) == 4, "Devrait avoir 4 lignes (1 supprimée)"

    def test_fill_nan_with_median(self, sample_data_with_nan):
        """Test remplissage des NaN avec médiane."""
        processor = DataProcessor()
        df_clean = processor.clean_data(sample_data_with_nan)

        # Vérifier absence de NaN dans features numériques
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Defective':
                assert df_clean[col].isnull().sum() == 0, \
                    f"Colonne {col} devrait avoir 0 NaN"

    def test_remove_duplicates(self, sample_data_with_duplicates):
        """Test suppression des doublons."""
        processor = DataProcessor()
        df_clean = processor.clean_data(sample_data_with_duplicates)

        assert len(df_clean) == 3, \
            "Devrait avoir 3 lignes uniques (1 doublon supprimé)"
        assert df_clean.duplicated().sum() == 0, \
            "Ne devrait pas avoir de doublons"

    def test_cleaning_preserves_data_types(self, sample_data):
        """Test que le nettoyage préserve les types de données."""
        processor = DataProcessor()
        df_clean = processor.clean_data(sample_data.copy())

        assert df_clean['LOC_TOTAL'].dtype in [np.int32, np.int64], \
            "LOC_TOTAL devrait rester entier"
        assert df_clean['HALSTEAD_VOLUME'].dtype in [np.float32, np.float64], \
            "HALSTEAD_VOLUME devrait rester float"

    def test_empty_dataframe(self):
        """Test avec DataFrame vide."""
        processor = DataProcessor()
        empty_df = pd.DataFrame()

        df_clean = processor.clean_data(empty_df)
        assert len(df_clean) == 0, "DataFrame vide devrait rester vide"


# ============================================
# TESTS DE SPLIT
# ============================================

@pytest.mark.unit
class TestDataSplitting:
    """Tests de la séparation train/test."""

    def test_split_ratio(self, sample_data):
        """Test que le ratio train/test est respecté."""
        processor = DataProcessor()
        X = sample_data.drop('Defective', axis=1)
        y = sample_data['Defective']

        X_train, X_test, y_train, y_test = processor.split_data(
            X, y, test_size=0.2
        )

        total = len(X_train) + len(X_test)
        assert len(X_test) / total == pytest.approx(0.2, abs=0.01), \
            "Test set devrait être ~20%"

    def test_stratification(self, sample_data):
        """Test que la stratification préserve la distribution."""
        processor = DataProcessor()
        X = sample_data.drop('Defective', axis=1)
        y = sample_data['Defective']

        X_train, X_test, y_train, y_test = processor.split_data(X, y)

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        assert train_ratio == pytest.approx(test_ratio, abs=0.05), \
            "Distribution train/test devrait être similaire"

    def test_split_reproducibility(self, sample_data):
        """Test que le split est reproductible avec même random_state."""
        processor = DataProcessor()
        X = sample_data.drop('Defective', axis=1)
        y = sample_data['Defective']

        # Premier split
        X_train1, X_test1, _, _ = processor.split_data(
            X, y, random_state=42
        )

        # Deuxième split avec même seed
        X_train2, X_test2, _, _ = processor.split_data(
            X, y, random_state=42
        )

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)

    def test_no_data_leakage(self, sample_data):
        """Test qu'il n'y a pas de fuite de données entre train/test."""
        processor = DataProcessor()
        X = sample_data.drop('Defective', axis=1)
        y = sample_data['Defective']

        X_train, X_test, _, _ = processor.split_data(X, y)

        # Vérifier qu'aucune ligne de test n'est dans train
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert len(train_indices.intersection(test_indices)) == 0, \
            "Pas de données communes entre train et test"


# ============================================
# TESTS D'INTÉGRATION
# ============================================

@pytest.mark.integration
class TestDataProcessorIntegration:
    """Tests d'intégration du pipeline complet."""

    def test_complete_pipeline(self, sample_csv_file):
        """Test du pipeline complet load -> clean -> split."""
        processor = DataProcessor(sample_csv_file)

        # 1. Load
        df = processor.load_data()
        assert df is not None

        # 2. Clean
        df_clean = processor.clean_data(df)
        assert df_clean['Defective'].isnull().sum() == 0

        # 3. Split
        X = df_clean.drop('Defective', axis=1)
        y = df_clean['Defective']
        X_train, X_test, y_train, y_test = processor.split_data(X, y)

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_pipeline_with_messy_data(self, sample_data_with_nan):
        """Test pipeline avec données sales."""
        processor = DataProcessor()

        # Ajouter des doublons et des infinis
        messy_data = sample_data_with_nan.copy()
        messy_data = pd.concat([messy_data, messy_data.iloc[:2]],
                               ignore_index=True)
        messy_data.loc[0, 'HALSTEAD_VOLUME'] = np.inf

        # Nettoyer
        df_clean = processor.clean_data(messy_data)

        # Vérifications
        assert df_clean['Defective'].isnull().sum() == 0
        assert not np.isinf(df_clean.select_dtypes(include=[np.number])).any().any()
        assert df_clean.duplicated().sum() == 0


# ============================================
# TESTS PARAMÉTRÉS
# ============================================

@pytest.mark.unit
@pytest.mark.parametrize("test_size,expected_min,expected_max", [
    (0.1, 8, 12),  # 10% de 100 = 10
    (0.2, 18, 22),  # 20% de 100 = 20
    (0.3, 28, 32),  # 30% de 100 = 30
])
def test_split_with_different_ratios(sample_data, test_size,
                                     expected_min, expected_max):
    """Test split avec différents ratios."""
    processor = DataProcessor()
    X = sample_data.drop('Defective', axis=1)
    y = sample_data['Defective']

    _, X_test, _, _ = processor.split_data(X, y, test_size=test_size)

    assert expected_min <= len(X_test) <= expected_max, \
        f"Test set devrait contenir {expected_min}-{expected_max} échantillons"


# ============================================
# TESTS DE PERFORMANCE
# ============================================

@pytest.mark.slow
def test_large_dataset_performance(benchmark):
    """Test de performance avec gros dataset."""
    # Créer un gros dataset
    np.random.seed(42)
    large_data = pd.DataFrame({
        'feature1': np.random.randn(100000),
        'feature2': np.random.randn(100000),
        'Defective': np.random.choice([0, 1], 100000)
    })

    processor = DataProcessor()

    # Benchmarker le nettoyage
    result = benchmark(processor.clean_data, large_data)

    assert len(result) > 0, "Devrait retourner des données"


# ============================================
# FIXTURES LOCALES
# ============================================

@pytest.fixture
def processor():
    """Fixture locale pour un processor."""
    return DataProcessor()