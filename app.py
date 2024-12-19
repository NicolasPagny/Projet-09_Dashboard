import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

# Charger les fichiers automatiquement
VAL_LGBM_FILE = "val/lgbm_val.pkl"  # Fichier des données
VAL_TABNET_FILE = "val/tabnet_val.pkl"  # Fichier des données
LGBM_MODEL_FILE = "models/lgbm_model.pkl"  # Modèle LightGBM
TABNET_MODEL_FILE = "models/tabnet_model.pkl"  # Modèle TabNet

### Les fonctions

def load_data(file_name, model_name):
    """
    Charge les données et affiche les premiers individus

    Args:
        file_name (str) : nom ou chemin du fichier 
        model_name (str): nom du modèle
    """
    try:
        data = pd.read_pickle(file_name)
        st.write(f"### Aperçu des données de validation prétraitées pour {model_name}")
        st.write(f"Dimensions : {data["X"].shape}")
        st.dataframe(data["X"].head())

        return data

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()

def predict(model, X, y): 
    """
    Récupérer les prédictions et la classification report

    Args:
        model (Model): le modèle à utiliser
        X (numpy): les features
        y (numpy): le terrain de vérité
    """
    predictions = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    report = classification_report(y, predictions, output_dict=True)

    return predictions, proba, report

def show_confusion_matrice(y, predictions, model_name):
    """
    Affiche la matrice de confusion
    
    Args:
        y (numpy): le terrain de vérité
        predictions (numpy): les prédictions obtenues
        model_name (str): le nom du modèle
    """

    st.write(f"### {model_name}")
    cm_tabnet = confusion_matrix(y, predictions)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm_tabnet, annot=True, fmt='d', cmap='Blues', xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"], ax=ax)
    st.pyplot(fig)

def show_features_importances(model, X, model_name):
    """
    Affiche une présentation de features importances globale

    Args:
        model (Model): le modèle à utiliser
        X (numpy): les features
        model_name: le nom du modèle
    """
    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    feature_importance_df = feature_importance_df[feature_importance_df["Importance"] > 0.001]
    feature_importance_df = feature_importance_df[: 10]

    # Bar plot des importances
    st.write(f"### {model_name}")
    fig = plt.figure(figsize=(10, 5))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Global Feature Importances {model_name}")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

def cost(seuils, pred_proba, y): 
    """
    Affiche une présentation du coût de perte par seuil

    seuils (range): le champ des seuils à tester
    pred_proba (numpy): les probabilité de prédictions
    y (numpy): le terrain de vérité

    """

    # Définir le ratio de coût entre le faux négatif et le faux positif
    ratio_cout_fn_fp = 10

    # Calculer les coûts totaux pour chaque seuil
    couts_totaux = []
    for seuil in seuils:
        # Classer les exemples en fonction du seuil
        predictions = (pred_proba >= seuil).astype(int)

        # Calculer la matrice de confusion
        conf_matrix = confusion_matrix(y, predictions)
        
        # Extraire les éléments de la matrice de confusion
        _, fp = conf_matrix[1, 1], conf_matrix[0, 1]
        _, fn = conf_matrix[0, 0], conf_matrix[1, 0]
        
        # Calculer le coût total en tenant compte de la matrice de coût
        cout_total = fp + fn * ratio_cout_fn_fp
        
        # Ajouter le coût total à la liste
        couts_totaux.append(cout_total)

    return couts_totaux

### Corps de code

# Titre du dashboard
st.title("Comparaison de Modèles : LGBM vs TABNET")

# Étape 1 : Chargement des données
# Charger les données
val_lgbm_data = load_data(VAL_LGBM_FILE, "LGBM")
val_tabnet_data = load_data(VAL_TABNET_FILE, "TABNET")

# Étape 2 : Chargement des modèles
try:
    load_lgbm_model = joblib.load(LGBM_MODEL_FILE)
    load_tabnet_model = joblib.load(TABNET_MODEL_FILE)
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# Étape 3 : Évaluation des modèles
st.write("## Résultats des modèles")

comparaison = {
    "Model": ["LGBM", "Tabnet"],
    "temps entraînements": [load_lgbm_model["training_time"], load_tabnet_model["training_time"]],
    "temps prédictions" : [load_lgbm_model["predict_time"], load_tabnet_model["predict_time"]]
}

st.table(pd.DataFrame(comparaison))

lgbm_model = load_lgbm_model["model"]
tabnet_model = load_tabnet_model["model"]
lgbm_predictions, lgbm_proba, lgbm_report = predict(lgbm_model, val_lgbm_data["X"], val_lgbm_data["y"])
tabnet_predictions, tabnet_proba, tabnet_report = predict(tabnet_model, val_tabnet_data["X"].values.astype("float32"), val_tabnet_data["y"])

col1, col2 = st.columns(2)
with col1:
    df_lgbm = pd.DataFrame(lgbm_report).transpose()
    st.write("### Rapport de classification LGBM")
    st.dataframe(df_lgbm.style.format("{:.2f}"))

with col2:
    df_tabnet = pd.DataFrame(tabnet_report).transpose()
    st.write("### Rapport de classification TABNET")
    st.dataframe(df_tabnet.style.format("{:.2f}"))

# Etape 4 : présentation des classifications report sous forme de graphique en barres

st.write("## Graphiques en barres du rapport de classification")

# Préparer les données pour la visualisation
metrics = ['precision', 'recall', 'f1-score']
classes = ["0.0", "1.0"]

data = {metric: [[lgbm_report[class_][metric], tabnet_report[class_][metric]] for class_ in classes] for metric in metrics}

# Création d'un graphique pour chaque classe
fig, axes = plt.subplots(1, len(classes), figsize=(12, 6), sharey=True)

for i, class_ in enumerate(classes):
    ax = axes[i]
    # Obtenir les valeurs pour chaque métrique et modèle
    values_model_1 = [data[metric][i][0] for metric in metrics]
    values_model_2 = [data[metric][i][1] for metric in metrics]

    # Positionnement des barres
    x = np.arange(len(metrics))
    width = 0.35  # Largeur des barres

    # Barres pour chaque modèle
    ax.bar(x - width / 2, values_model_1, width, label='LGBM', color='skyblue')
    ax.bar(x + width / 2, values_model_2, width, label='TABNET', color='orange')

    # Titres et légendes
    ax.set_title(f"Classe {class_}", fontsize=14)
    ax.set_xlabel("Métriques", fontsize=12)
    if i == 0:
        ax.set_ylabel("Valeur", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.bar_label(ax.containers[0], fmt="%.2f")  # Modèle 1
    ax.bar_label(ax.containers[1], fmt="%.2f")  # Modèle 2

plt.tight_layout()
st.pyplot(fig)


# Étape 4 : Visualisation des courbes ROC
st.write("## Courbes ROC")
fig, ax = plt.subplots(figsize=(5,10))
RocCurveDisplay.from_predictions(val_lgbm_data["y"], lgbm_proba, ax=ax, name="LGBM")
RocCurveDisplay.from_predictions(val_tabnet_data["y"], tabnet_proba, ax=ax, name="TABNET")
ax.set_xlabel("FPR", fontsize=5)
ax.set_ylabel("TPR", fontsize=5)
ax.set_title("courbes ROC", fontsize=5)
ax.tick_params(labelsize=5)
ax.legend(fontsize=5)
st.pyplot(fig)

# etape : coûts
st.write("## Graphique de coûts par seuil")

# Définir une liste de seuils à tester
seuils = np.arange(0, 1.05, 0.05)

# calcul des coûts
lgbm_cost = cost(seuils, lgbm_proba, val_lgbm_data["y"])
tabnet_cost = cost(seuils, tabnet_proba, val_tabnet_data["y"])

# Trouver le seuil qui minimise le coût total
seuil_optimal_lgbm = seuils[np.argmin(lgbm_cost)]
cout_minimum_lgbm = np.min(lgbm_cost)
seuil_optimal_tabnet = seuils[np.argmin(tabnet_cost)]
cout_minimum_tabnet = np.min(tabnet_cost)

#Convertir les listes en tableaux numpy pour la facilité de manipulation
seuils = np.array(seuils)
lgbm_cost = np.array(lgbm_cost)
tabnet_cost = np.array(tabnet_cost)

# Tracer les deux courbes sur un seul graphique
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(seuils, lgbm_cost, marker='o', color='blue', label='LightGBM')
ax.plot(seuils, tabnet_cost, marker='o', color='orange', label='TabNet')

# Ajouter les titres, légendes et grilles
ax.set_title('Coût total en fonction du seuil', fontsize=16)
ax.set_xlabel('Seuil', fontsize=14)
ax.set_ylabel('Coût total', fontsize=14)
ax.grid(True)
ax.legend(fontsize=12)

# Afficher le graphique dans Streamlit
st.pyplot(fig)

cost_datas = {
    "Seuil optimal": [seuil_optimal_lgbm, seuil_optimal_tabnet],
    "Coût Minimum": [cout_minimum_lgbm, cout_minimum_tabnet]
}

cost_df = pd.DataFrame(cost_datas, index=["LGBM", "TABNET"])
st.dataframe(cost_df)

# Étape 5 : Matrices de confusion
st.write("### Matrices de confusion")

col1, col2 = st.columns(2)

with col1:
    show_confusion_matrice(val_lgbm_data["y"], lgbm_predictions, "LGBM") 

with col2:
    show_confusion_matrice(val_tabnet_data["y"], tabnet_predictions, "TABNET") 

# global features importances
st.write("### Feature importance Global")
show_features_importances(lgbm_model, val_lgbm_data["X"], "LGBM")
show_features_importances(tabnet_model, val_tabnet_data["X"], "TABNET")

# Choisir une observation à expliquer (par exemple, la première)
st.write("### Observation à expliquer")
observation_index = st.slider("Choisis l'index de l'observation", 0, len(val_lgbm_data["X"]) - 1, 0)

# Afficher les explications
st.write("### Feature Importance Locale")


# Partie LGBM
st.write("#### LGBM")

# local feature importance
features = val_lgbm_data["X"]
observation = features.iloc[observation_index]
st.write("Observation :")
st.write(observation)

explainer = LimeTabularExplainer(
    training_data=features.values,
    feature_names=features.columns,
    class_names=["Classe 0", "Classe 1"],
    mode="classification"
)

# Obtenir les explications locales avec LIME
explanation = explainer.explain_instance(
    data_row=observation.values,
    predict_fn=lgbm_model.predict_proba
)

exp_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Contribution"])

# Visualisation graphique des explications
prediction = lgbm_model.predict_proba([observation.values])[0]
st.write("Prédictions :")
st.write(f"Classe 0 : {prediction[0]:.2f}, Classe 1 : {prediction[1]:.2f}")

# Explication locale
fig = explanation.as_pyplot_figure()
st.pyplot(fig)

# Partie Tabnet
st.write("#### TabNet")

explain_matrix, masks = tabnet_model.explain(val_tabnet_data["X"].values.astype("float32"))

# local feature importance
features = val_tabnet_data["X"]
observation = features.iloc[observation_index]
st.write("Observation :")
st.write(observation)

# Visualisation graphique des explications
prediction = tabnet_model.predict_proba([observation.values.astype("float32")])[0]
st.write("Prédictions :")
st.write(f"Classe 0 : {prediction[0]:.2f}, Classe 1 : {prediction[1]:.2f}")

# Contribution des features pour un exemple spécifique
contributions = explain_matrix[observation_index]

# Affichage des contributions par feature
feature_contributions = pd.DataFrame({
    "Feature": val_tabnet_data["X"].columns, 
    "Contribution": contributions
}).sort_values(by="Contribution", ascending=False)

feature_contributions = feature_contributions[feature_contributions["Contribution"] > 0]

# Bar plot des importances
fig = plt.figure(figsize=(10, 5))
plt.barh(feature_contributions["Feature"], feature_contributions["Contribution"])
plt.xlabel("Contribution")
plt.ylabel("Feature")
plt.title("Local Feature Importances for idx " + str(observation_index))
plt.gca().invert_yaxis()
st.pyplot(fig)