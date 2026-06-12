import matplotlib.pyplot as plt

modeli = ['RF tuned', 'RF default', 'Moj RF', 'XGBoost', 'Linear\nRegression']
r2_scores = [0.6384, 0.6304, 0.6288, 0.5949, 0.4702]

# Boje - istakni tvoj custom model
boje = ['steelblue', 'steelblue', 'seagreen', 'steelblue', 'coral']

plt.figure(figsize=(11, 6))
bars = plt.bar(modeli, r2_scores, color=boje, edgecolor='black', alpha=0.85)

plt.ylabel('R² score', fontsize=12)
plt.title('Poređenje modela po R² score', fontsize=14, fontweight='bold')
plt.ylim(0, 0.8)

# Vrednosti iznad svake kolone
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, score + 0.015, 
             f'{score:.4f}', ha='center', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('poredjenje_modela.png', dpi=150)
plt.show()