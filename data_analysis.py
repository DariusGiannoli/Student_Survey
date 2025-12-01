import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# 1. Chargement & Préparation
try:
    df = pd.read_csv("TDS_HW4_Final_Dataset.csv")
except FileNotFoundError:
    # Fallback si tu as utilisé l'autre nom
    df = pd.read_csv("documents/answers_form.csv")

df['Is Master'] = df['Study Level'].apply(lambda x: 'Master' if 'Master' in x else 'Bachelor')

# --- CONFIGURATION STYLE ---
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

# Couleurs inspirées de tes images
colors_levels = ['#3498db', '#e74c3c'] # Bleu (Bachelor) vs Rouge (Master)

print("--- 1. ANALYSE STATISTIQUE ---")
contingency_table = pd.crosstab(df['Is Master'], df['Has Conflict'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2: {chi2:.2f}, p-value: {p:.4e}")

print("\n--- 2. GÉNÉRATION DES GRAPHIQUES ---")

# --- FIGURE 1: Sample Distribution (Barres Horizontales) ---
plt.figure(figsize=(12, 7))
sec_counts = df['Section'].value_counts(normalize=True) * 100

ax = sns.barplot(x=sec_counts.values, y=sec_counts.index, palette="viridis")
plt.title('Sample Distribution by Section')
plt.xlabel('Percentage (%)')

# MODIFICATION ICI : Axe de 0 à 100
plt.xlim(0, 100)

for i, p in enumerate(ax.patches):
    width = p.get_width()
    # Ajout d'un petit offset (+1) pour que le texte ne colle pas à la barre
    ax.text(width + 1, p.get_y() + p.get_height()/2, f'{width:.1f}%',
            va='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig("Figure1_Sample_Distribution_Styled.png")
print("-> Figure 1 (Distribution 9 sections) générée.")


# --- FIGURE 2: Evolution by Year ---
plt.figure(figsize=(10, 6))
order = ['BA1', 'BA3', 'BA5', 'Master']
rates = df.groupby('Study Level')['Has Conflict'].apply(lambda x: (x=='Yes').mean()*100).reindex(order)
sns.lineplot(x=rates.index, y=rates.values, marker='o', markersize=10, linewidth=4, color='#8e44ad')
plt.title('Evolution of Conflict Probability')
plt.ylabel('Conflict Rate (%)')
plt.xlabel('')
plt.ylim(0, 100)
for i, val in enumerate(rates.values):
    plt.text(i, val + 3, f'{val:.1f}%', ha='center', fontweight='bold', color='#8e44ad')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Figure2_Evolution_Styled.png")
print("-> Figure 2 générée.")


# --- FIGURE 3: The Paradox ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# A. Prévalence
conflict_rates = df.groupby('Is Master')['Has Conflict'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
sns.barplot(data=conflict_rates, x='Is Master', y='Has Conflict', ax=axes[0], palette=colors_levels, order=['Bachelor', 'Master'])
axes[0].set_title("Conflict Frequency")
axes[0].set_ylabel("Rate (%)")
axes[0].set_ylim(0, 100)
axes[0].set_xlabel("")
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.1f}%', (p.get_x()+0.4, p.get_height()), ha='center', va='bottom', fontweight='bold', fontsize=12)

# B. Sentiment
df_conf = df[df['Has Conflict']=='Yes']
care_rates = df_conf.groupby('Is Master')['Do you care?'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
sns.barplot(data=care_rates, x='Is Master', y='Do you care?', ax=axes[1], palette=colors_levels, order=['Bachelor', 'Master'])
axes[1].set_title("Sensitivity (% Who Care)")
axes[1].set_ylabel("")
axes[1].set_ylim(0, 100)
axes[1].set_xlabel("")
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.1f}%', (p.get_x()+0.4, p.get_height()), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("Figure3_Paradox_Styled.png")
print("-> Figure 3 générée.")


# --- FIGURE 4: Drivers ---
plt.figure(figsize=(12, 7))
counts = df_conf.groupby(['Is Master', 'Reason']).size().reset_index(name='Count')
totals = df_conf.groupby('Is Master')['Reason'].count().reset_index(name='Total')
counts = counts.merge(totals, on='Is Master')
counts['Percentage'] = (counts['Count'] / counts['Total']) * 100

order_x = ["Academic Interest", "Curriculum Requirement", "Best Schedule Fit", "Limited Alternatives"]

ax = sns.barplot(data=counts, x='Reason', y='Percentage', hue='Is Master',
            palette=colors_levels, hue_order=['Bachelor', 'Master'], order=order_x)

plt.title('Decision Drivers: Why choose a conflict?')
plt.ylabel('Percentage (%)')
plt.xlabel('')
plt.ylim(0, 100)
plt.legend(title='')

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x()+p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig("Figure4_Drivers_Styled.png")
print("-> Figure 4 générée.")


# --- FIGURE IC EXCEPTION ---
plt.figure(figsize=(8, 6))
df_ba3 = df[df['Study Level'] == 'BA3'].copy()
# Définir le groupe IC vs Autres
ic_sections = ['Computer Science', 'Communication Systems']
df_ba3['Group'] = df_ba3['Section'].apply(lambda x: 'IC Section' if x in ic_sections else 'Other Sections')
rates_ic = df_ba3.groupby('Group')['Has Conflict'].apply(lambda x: (x=='Yes').mean()*100).reset_index()

ax = sns.barplot(x='Group', y='Has Conflict', data=rates_ic, palette=['#e74c3c', '#95a5a6'], order=['IC Section', 'Other Sections'])
plt.title("Structural Impact: IC vs Others (BA3)")
plt.ylabel("Conflict Rate (%)")
plt.xlabel("")
plt.ylim(0, 50)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x()+0.4, p.get_height()), ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("Figure_IC_Exception_Styled.png")
print("-> Figure IC Exception générée.")

print("\nTerminé.")
