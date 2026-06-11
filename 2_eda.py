import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/cleaned_data.csv')

# 1. Korelaciona matrica
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korelaciona matrica')
plt.tight_layout()
plt.savefig('data/EDA_grafikoni/korelacija.png')
plt.show()

# 2. Boxplotovi za numeričke kolone
num_cols = ['avg_rain', 'Nitrogen', 'POTASH', 'PHOSPHATE', 'RICE_YIELD']

plt.figure(figsize=(14, 6))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.savefig('data/EDA_grafikoni/boxplots_drugi_put.png')
plt.show()