import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import os

print("=" * 70)
print("📊 GÉNÉRATION DES VISUALISATIONS")
print("=" * 70)

# Configuration style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Créer dossier
figures_dir = r'C:\Users\M9-electro\Desktop\bug-predictor\results\figures'
os.makedirs(figures_dir, exist_ok=True)

# 1. CHARGER DONNÉES
df = pd.read_csv(r'C:\Users\M9-electro\Desktop\bug-predictor\data\processed\data.csv')
df = df.dropna(subset=['Defective'])

mapping = {0: 0, 1: 1, 0.0: 0, 1.0: 1, 'N': 0, 'Y': 1}
df['Defective'] = df['Defective'].map(mapping).astype(int)

X = df.drop(columns=['Defective', 'source', 'label'], errors='ignore')
y = df['Defective']

X = X.fillna(X.median()).replace([np.inf, -np.inf], np.nan).fillna(X.median())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Charger le meilleur modèle
model_dir = r'C:\Users\M9-electro\Desktop\bug-predictor\models'
model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))

# Prédictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ========================================
# FIGURE 1 : DISTRIBUTION DU DATASET
# ========================================
print("\n📊 Figure 1: Distribution du dataset...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 Distribution des bugs
ax = axes[0, 0]
bug_counts = y.value_counts()
colors = ['#2ecc71', '#e74c3c']
ax.bar(['No Bug', 'Bug'], bug_counts.values, color=colors, alpha=0.7)
ax.set_title('Distribution des Bugs dans le Dataset', fontsize=14, fontweight='bold')
ax.set_ylabel('Nombre de fichiers')
for i, v in enumerate(bug_counts.values):
    ax.text(i, v + 50, f'{v}\n({v/len(y)*100:.1f}%)', ha='center', fontweight='bold')

# 1.2 Distribution par source
ax = axes[0, 1]
if 'source' in df.columns:
    source_counts = df['source'].value_counts()
    ax.barh(source_counts.index, source_counts.values, color='steelblue', alpha=0.7)
    ax.set_title('Distribution par Source (Dataset)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Nombre de fichiers')
else:
    ax.text(0.5, 0.5, 'Source non disponible', ha='center', va='center')
    ax.set_title('Distribution par Source')

# 1.3 Top 10 Features Importance
ax = axes[1, 0]
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='coral', alpha=0.7)
ax.set_title('Top 10 Features les Plus Importantes', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
ax.invert_yaxis()

# 1.4 Bugs par percentile de complexité
ax = axes[1, 1]
if 'CYCLOMATIC_COMPLEXITY' in X.columns:
    complexity = df['CYCLOMATIC_COMPLEXITY'].fillna(0)
    percentiles = pd.qcut(complexity, q=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    bug_by_complexity = df.groupby(percentiles)['Defective'].mean() * 100
    ax.bar(bug_by_complexity.index, bug_by_complexity.values, color='purple', alpha=0.7)
    ax.set_title('Pourcentage de Bugs par Complexité', fontsize=14, fontweight='bold')
    ax.set_ylabel('% de Bugs')
    ax.set_xlabel('Percentile de Complexité')
else:
    ax.text(0.5, 0.5, 'Complexité non disponible', ha='center', va='center')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '01_dataset_distribution.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Sauvegardé: 01_dataset_distribution.png")
plt.close()

# ========================================
# FIGURE 2 : PERFORMANCES DU MODÈLE
# ========================================
print("\n📊 Figure 2: Performances du modèle...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2.1 Matrice de confusion
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
ax.set_ylabel('Réalité')
ax.set_xlabel('Prédiction')
ax.set_xticklabels(['No Bug', 'Bug'])
ax.set_yticklabels(['No Bug', 'Bug'])

# 2.2 ROC Curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# 2.3 Distribution des probabilités
ax = axes[1, 0]
ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='No Bug', color='green')
ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='Bug', color='red')
ax.set_xlabel('Probabilité prédite de Bug')
ax.set_ylabel('Fréquence')
ax.set_title('Distribution des Probabilités de Prédiction', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2.4 Métriques
ax = axes[1, 1]
metrics_df = pd.read_csv(os.path.join(model_dir, 'metrics_comparison.csv'))
metrics_df_plot = metrics_df.set_index('Metric')

x = np.arange(len(metrics_df_plot.index))
width = 0.35

bars1 = ax.bar(x - width/2, metrics_df_plot['Train'] * 100, width, label='Train', color='skyblue', alpha=0.8)
bars2 = ax.bar(x + width/2, metrics_df_plot['Test'] * 100, width, label='Test', color='coral', alpha=0.8)

ax.set_ylabel('Score (%)')
ax.set_title('Comparaison Train vs Test', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df_plot.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=70, color='r', linestyle='--', linewidth=1, label='Objectif 70%')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '02_model_performance.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Sauvegardé: 02_model_performance.png")
plt.close()

# ========================================
# FIGURE 3 : COMPARAISON DES MODÈLES
# ========================================
print("\n📊 Figure 3: Comparaison des modèles...")

comparison_df = pd.read_csv(os.path.join(model_dir, 'model_comparison.csv'))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 Accuracy
ax = axes[0, 0]
sns.barplot(data=comparison_df, y='Modèle', x='Accuracy', ax=ax, palette='viridis')
ax.set_title('Accuracy par Modèle', fontsize=14, fontweight='bold')
ax.axvline(x=70, color='r', linestyle='--', linewidth=1, label='Objectif 70%')
ax.legend()

# 3.2 Recall
ax = axes[0, 1]
sns.barplot(data=comparison_df, y='Modèle', x='Recall', ax=ax, palette='coolwarm')
ax.set_title('Recall par Modèle', fontsize=14, fontweight='bold')
ax.axvline(x=50, color='orange', linestyle='--', linewidth=1, label='Cible 50%')
ax.legend()

# 3.3 F1-Score
ax = axes[1, 0]
sns.barplot(data=comparison_df, y='Modèle', x='F1-Score', ax=ax, palette='plasma')
ax.set_title('F1-Score par Modèle', fontsize=14, fontweight='bold')

# 3.4 Temps d'entraînement
ax = axes[1, 1]
sns.barplot(data=comparison_df, y='Modèle', x='Temps (s)', ax=ax, palette='magma')
ax.set_title('Temps d\'Entraînement (secondes)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '03_model_comparison.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Sauvegardé: 03_model_comparison.png")
plt.close()

print("\n" + "="*70)
print("✅ TOUTES LES VISUALISATIONS GÉNÉRÉES")
print("="*70)
print(f"\n📁 Dossier: {figures_dir}")
print("   - 01_dataset_distribution.png")
print("   - 02_model_performance.png")
print("   - 03_model_comparison.png")