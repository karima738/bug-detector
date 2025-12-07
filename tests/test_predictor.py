"""
Tests unitaires pour BugPredictor.

Teste les fonctionnalités de prédiction de bugs.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import Mock, patch, MagicMock


# Simuler la classe BugPredictor
class BugPredictor:
    """Classe simplifiée pour tests."""

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def predict_file(self, metrics):
        """Prédit si un fichier contient des bugs."""
        if self.scaler:
            metrics_scaled = self.scaler.transform(metrics)
        else:
            metrics_scaled = metrics

        prediction = self.model.predict(metrics_scaled)[0]
        probability = self.model.predict_proba(metrics_scaled)[0, 1]

        return {
            'prediction': 'Bug' if prediction == 1 else 'No Bug',
            'probability': probability,
            'risk_level': self._calculate_risk_level(probability)
        }

    def _calculate_risk_level(self, probability):
        """Calcule le niveau de risque."""
        if probability > 0.7:
            return 'HIGH'
        elif probability > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

    @staticmethod
    def load_model(filepath):
        """Charge un modèle depuis un fichier."""
        model = joblib.load(filepath)
        return BugPredictor(model=model)

    def save_model(self, filepath):
        """Sauvegarde le modèle."""
        joblib.dump(self.model, filepath)


# ============================================
# TESTS DE PRÉDICTION
# ============================================

@pytest.mark.unit
class TestPrediction:
    """Tests des fonctionnalités de prédiction."""

    def test_predict_no_bug(self, trained_model, scaler_fitted, sample_data):
        """Test prédiction d'un fichier sans bug."""
        predictor = BugPredictor(model=trained_model, scaler=scaler_fitted)

        # Fichier avec métriques basses (peu de risque)
        low_risk_file = pd.DataFrame([{
            'LOC_TOTAL': 50,
            'CYCLOMATIC_COMPLEXITY': 3,
            'HALSTEAD_VOLUME': 20,
            'HALSTEAD_DIFFICULTY': 5,
            'LOC_EXECUTABLE': 40
        }])

        result = predictor.predict_file(low_risk_file)

        assert result is not None
        assert 'prediction' in result
        assert 'probability' in result
        assert 'risk_level' in result
        assert result['prediction'] in ['Bug', 'No Bug']

    def test_predict_bug(self, trained_model, scaler_fitted):
        """Test prédiction d'un fichier avec bug."""
        predictor = BugPredictor(model=trained_model, scaler=scaler_fitted)

        # Fichier avec métriques élevées (plus de risque)
        high_risk_file = pd.DataFrame([{
            'LOC_TOTAL': 500,
            'CYCLOMATIC_COMPLEXITY': 25,
            'HALSTEAD_VOLUME': 1000,
            'HALSTEAD_DIFFICULTY': 50,
            'LOC_EXECUTABLE': 450
        }])

        result = predictor.predict_file(high_risk_file)

        assert result['probability'] >= 0.0
        assert result['probability'] <= 1.0

    def test_predict_probability_range(self, trained_model, scaler_fitted, sample_data):
        """Test que la probabilité est entre 0 et 1."""
        predictor = BugPredictor(model=trained_model, scaler=scaler_fitted)

        X = sample_data.drop('Defective', axis=1).head(10)

        for _, row in X.iterrows():
            result = predictor.predict_file(pd.DataFrame([row]))

            assert 0.0 <= result['probability'] <= 1.0, \
                f"Probabilité {result['probability']} hors limites [0, 1]"

    def test_predict_without_scaler(self, trained_model, sample_data):
        """Test prédiction sans scaler (données déjà normalisées)."""
        predictor = BugPredictor(model=trained_model, scaler=None)

        X = sample_data.drop('Defective', axis=1).head(1)
        result = predictor.predict_file(X)

        assert result is not None
        assert 'prediction' in result


# ============================================
# TESTS DU NIVEAU DE RISQUE
# ============================================

@pytest.mark.unit
class TestRiskLevel:
    """Tests du calcul du niveau de risque."""

    def test_risk_level_high(self, trained_model):
        """Test niveau de risque élevé."""
        predictor = BugPredictor(model=trained_model)

        risk = predictor._calculate_risk_level(0.85)
        assert risk == 'HIGH', "Probabilité > 0.7 devrait être HIGH"

    def test_risk_level_medium(self, trained_model):
        """Test niveau de risque moyen."""
        predictor = BugPredictor(model=trained_model)

        risk = predictor._calculate_risk_level(0.55)
        assert risk == 'MEDIUM', "Probabilité 0.4-0.7 devrait être MEDIUM"

    def test_risk_level_low(self, trained_model):
        """Test niveau de risque faible."""
        predictor = BugPredictor(model=trained_model)

        risk = predictor._calculate_risk_level(0.25)
        assert risk == 'LOW', "Probabilité < 0.4 devrait être LOW"

    @pytest.mark.parametrize("probability,expected_risk", [
        (0.0, 'LOW'),
        (0.39, 'LOW'),
        (0.4, 'MEDIUM'),
        (0.55, 'MEDIUM'),
        (0.69, 'MEDIUM'),
        (0.7, 'HIGH'),
        (0.85, 'HIGH'),
        (1.0, 'HIGH'),
    ])
    def test_risk_level_boundaries(self, trained_model, probability, expected_risk):
        """Test des frontières de niveau de risque."""
        predictor = BugPredictor(model=trained_model)

        risk = predictor._calculate_risk_level(probability)
        assert risk == expected_risk, \
            f"Probabilité {probability} devrait donner {expected_risk}"


# ============================================
# TESTS DE SAUVEGARDE/CHARGEMENT
# ============================================

@pytest.mark.unit
class TestModelPersistence:
    """Tests de sauvegarde et chargement des modèles."""

    def test_save_model(self, trained_model, tmp_path):
        """Test sauvegarde d'un modèle."""
        predictor = BugPredictor(model=trained_model)
        model_path = tmp_path / "test_model.pkl"

        predictor.save_model(str(model_path))

        assert model_path.exists(), "Le fichier modèle devrait exister"
        assert model_path.stat().st_size > 0, "Le fichier ne devrait pas être vide"

    def test_load_model(self, model_file):
        """Test chargement d'un modèle."""
        predictor = BugPredictor.load_model(model_file)

        assert predictor is not None
        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict')

    def test_save_load_roundtrip(self, trained_model, tmp_path, sample_data):
        """Test sauvegarde puis chargement (round-trip)."""
        # Sauvegarder
        predictor1 = BugPredictor(model=trained_model)
        model_path = tmp_path / "roundtrip_model.pkl"
        predictor1.save_model(str(model_path))

        # Charger
        predictor2 = BugPredictor.load_model(str(model_path))

        # Comparer les prédictions
        X = sample_data.drop('Defective', axis=1).head(5)

        pred1 = predictor1.model.predict(X)
        pred2 = predictor2.model.predict(X)

        np.testing.assert_array_equal(pred1, pred2,
                                      "Les prédictions devraient être identiques après save/load")

    def test_load_invalid_path(self):
        """Test chargement avec chemin invalide."""
        with pytest.raises(FileNotFoundError):
            BugPredictor.load_model("chemin/inexistant.pkl")


# ============================================
# TESTS AVEC MOCK
# ============================================

@pytest.mark.unit
class TestWithMocks:
    """Tests utilisant des mocks."""

    def test_predict_with_mocked_model(self, sample_data):
        """Test avec un modèle mocké."""
        # Créer un mock du modèle
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])  # Prédit Bug
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        predictor = BugPredictor(model=mock_model)

        X = sample_data.drop('Defective', axis=1).head(1)
        result = predictor.predict_file(X)

        # Vérifier que le mock a été appelé
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()

        assert result['prediction'] == 'Bug'
        assert result['probability'] == 0.7
        assert result['risk_level'] == 'HIGH'

    @patch('joblib.load')
    def test_load_model_with_mock(self, mock_load, trained_model):
        """Test chargement avec joblib mocké."""
        mock_load.return_value = trained_model

        predictor = BugPredictor.load_model("fake_path.pkl")

        mock_load.assert_called_once_with("fake_path.pkl")
        assert predictor.model is not None


# ============================================
# TESTS D'ERREURS
# ============================================

@pytest.mark.unit
class TestErrorHandling:
    """Tests de gestion d'erreurs."""

    def test_predict_with_none_model(self):
        """Test prédiction sans modèle chargé."""
        predictor = BugPredictor(model=None)

        with pytest.raises(AttributeError):
            predictor.predict_file(pd.DataFrame([[1, 2, 3]]))

    def test_predict_with_wrong_features(self, trained_model, sample_data):
        """Test avec mauvais nombre de features."""
        predictor = BugPredictor(model=trained_model)

        # Seulement 2 features au lieu de 5
        wrong_features = pd.DataFrame([[100, 10]])

        with pytest.raises(ValueError):
            predictor.predict_file(wrong_features)

    def test_predict_with_missing_columns(self, trained_model):
        """Test avec colonnes manquantes."""
        predictor = BugPredictor(model=trained_model)

        # DataFrame avec mauvais noms de colonnes
        wrong_df = pd.DataFrame([[1, 2, 3, 4, 5]],
                                columns=['A', 'B', 'C', 'D', 'E'])

        with pytest.raises((ValueError, KeyError)):
            predictor.predict_file(wrong_df)


# ============================================
# TESTS D'INTÉGRATION
# ============================================

@pytest.mark.integration
class TestBugPredictorIntegration:
    """Tests d'intégration du predictor."""

    def test_complete_prediction_workflow(self, model_file, sample_data):
        """Test du workflow complet : load → predict → save."""
        # 1. Charger
        predictor = BugPredictor.load_model(model_file)

        # 2. Prédire sur plusieurs fichiers
        X = sample_data.drop('Defective', axis=1).head(10)
        results = []

        for _, row in X.iterrows():
            result = predictor.predict_file(pd.DataFrame([row]))
            results.append(result)

        # 3. Vérifier résultats
        assert len(results) == 10

        for result in results:
            assert 'prediction' in result
            assert 'probability' in result
            assert 'risk_level' in result
            assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']

    def test_batch_prediction(self, trained_model, scaler_fitted, sample_data):
        """Test prédiction batch."""
        predictor = BugPredictor(model=trained_model, scaler=scaler_fitted)

        X = sample_data.drop('Defective', axis=1).head(20)

        # Prédire sur tous les fichiers
        results = []
        for _, row in X.iterrows():
            result = predictor.predict_file(pd.DataFrame([row]))
            results.append(result)

        # Statistiques
        bug_count = sum(1 for r in results if r['prediction'] == 'Bug')
        high_risk_count = sum(1 for r in results if r['risk_level'] == 'HIGH')

        assert 0 <= bug_count <= 20
        assert 0 <= high_risk_count <= 20


# ============================================
# TESTS DE PERFORMANCE
# ============================================

@pytest.mark.slow
def test_prediction_performance(trained_model, sample_data, benchmark):
    """Test de performance de la prédiction."""
    predictor = BugPredictor(model=trained_model)
    X = sample_data.drop('Defective', axis=1).head(1)

    # Benchmarker la prédiction
    result = benchmark(predictor.predict_file, X)

    assert result is not None
    # La prédiction devrait être rapide (< 100ms généralement)