import pandas as pd
from io import StringIO

# Charger le ground truth
df = pd.read_excel('evaluation/ground_truth.csv') if 'xlsx' in 'ground_truth.csv' else pd.read_csv('evaluation/ground_truth.csv', encoding='utf-8-sig')

total = len(df)
identifies = len(df[df['statut'] == 'identifie'])
erreurs = len(df[df['statut'] == 'erreur'])

VP = identifies
FN = erreurs
FP = 0

precision = VP / (VP + FP) if (VP + FP) > 0 else 0
rappel = VP / (VP + FN) if (VP + FN) > 0 else 0
f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0
accuracy = VP / total

faible = len(df[(df['statut']=='identifie') & (df['score'] < 0.5)])
moyen  = len(df[(df['statut']=='identifie') & (df['score'] >= 0.5) & (df['score'] < 0.7)])
eleve  = len(df[(df['statut']=='identifie') & (df['score'] >= 0.7)])

print("=" * 45)
print("  METRIQUES EVALUATION --- MedBox API")
print("=" * 45)
print(f"Dataset de test    : {total} images")
print(f"Identifies (VP)    : {VP}")
print(f"Non identifies(FN) : {FN}")
print(f"\n--- Metriques globales ---")
print(f"Precision          : {precision*100:.1f}%")
print(f"Rappel             : {rappel*100:.1f}%")
print(f"F1-Score           : {f1*100:.1f}%")
print(f"Accuracy           : {accuracy*100:.1f}%")
print(f"\n--- Distribution des scores ---")
print(f"Score faible (<0.5)   : {faible} ({100*faible/total:.1f}%)")
print(f"Score moyen (0.5-0.7) : {moyen} ({100*moyen/total:.1f}%)")
print(f"Score eleve (>0.7)    : {eleve} ({100*eleve/total:.1f}%)")
print(f"\nScore moyen : {df[df['statut']=='identifie']['score'].mean():.3f}")
print("=" * 45)

# Sauvegarder le rapport
with open('evaluation/metrics_report.txt', 'w') as f:
    f.write(f"Dataset de test : {total} images\n")
    f.write(f"Identifies      : {VP}\n")
    f.write(f"Non identifies  : {FN}\n")
    f.write(f"Precision       : {precision*100:.1f}%\n")
    f.write(f"Rappel          : {rappel*100:.1f}%\n")
    f.write(f"F1-Score        : {f1*100:.1f}%\n")
    f.write(f"Accuracy        : {accuracy*100:.1f}%\n")
    f.write(f"Score moyen     : {df[df['statut']=='identifie']['score'].mean():.3f}\n")

print("\nRapport sauvegarde : evaluation/metrics_report.txt")