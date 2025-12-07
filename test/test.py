import pandas as pd
import os

# --- 1. Charger ton dataset d'entraînement ---
# Remplace le chemin ci-dessous par ton vrai fichier d'entraînement
train_path = r"C:\Users\M9-electro\Desktop\bug-predictor\data\processed\data.csv"
df = pd.read_csv(train_path)

# --- 2. Sélectionner un échantillon pour le test (20% par défaut) ---
test_df = df.sample(frac=0.2, random_state=42)

# --- 3. Chemin de sortie du fichier test.csv ---
output_dir = r"/test"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "test.csv")

# --- 4. Sauvegarder le fichier ---
test_df.to_csv(output_path, index=False)

print("test.csv généré avec succès dans :", output_path)
