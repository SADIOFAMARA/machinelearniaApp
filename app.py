import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# Configuration de l'affichage du local host, du message sur la barre de navigation de la page web.
st.set_page_config(page_title="Data Visualization & Prediction App", layout="wide")
# Configuration d'un thème global


# Configuration du panneau latéral gauche, la barre latérale gauche
def main():
    # Sidebar (Barre latérale gauche)
    st.sidebar.title("Sections")
    page = st.sidebar.radio("Navigation", ["Accueil", "Présentation", "Visualisation", "Prédiction"])

    # Code pour permettre à l'utilisateur d'importer une base de données
    uploaded_file = st.sidebar.file_uploader("Importer une base de données (CSV)", type=["csv"])

    # Activation de la base de données et navigation dans les différentes pages
    if page == "Accueil":
        home_page()
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if page == "Présentation":
            presentation_page(df)
        elif page == "Visualisation":
            visualization_page(df)
        elif page == "Prédiction":
            prediction_page(df)
    else:
        # Message qui s'affiche dans les autres pages au cas ou la base de données n'a pas été importée au niveau de la page d'acceuille
        st.sidebar.warning("Veuillez télécharger une base de données pour commencer.")
        
# Configuration de la page d'accueille
def home_page():
    st.title("Prédiction de la performance des étudiants")
    st.image("https://cdn.pixabay.com/photo/2017/09/26/04/28/classroom-2787754_1280.jpg", caption="Pixabay",  use_container_width=True)
    st.write("""
        **Contexte**
        
        Dans un environnement éducatif de plus en plus axé sur les données, il est essentiel pour les établissements scolaires de comprendre les 
        facteurs qui influencent la performance académique des étudiants.

        Les performances académiques des étudiants peuvent être influencées par divers facteurs, notamment leur milieu socio-économique, leur engagement,
        et les ressources disponibles. Analyser ces facteurs peut fournir des insights précieux pour les éducateurs et les décideurs afin d’améliorer les résultats scolaires.

        L’objectif principal est de prédire les résultats scolaires des étudiants en fonction de leurs caractéristiques, afin d’identifier les étudiants à risque et d’aider 
        les éducateurs à intervenir de manière proactive.


         **Présentation de l'application**
             
         Cette application
        
        - Présente les données nécessaires pour réaliser la prédiction de la performance des étudiants ; 
        - Présente les tendances de toutes les variables à travers des graphiques univriés, bivariés ; 
        - Permet de prédire à l'aide d'un modèle adapté la performance des étudiants ; 
        - Fournit une description numérique de toutes les variables ; 

        **Auteurs :**
        - **Famara SADIO**, *Elève Ingénieur Statisticien Economiste en 2ème année*
        - **Saran NDIAYE**, *Elève Ingénieur Statisticien Economiste en 2ème année*
        - **Amadou YOUM**,  *Elève Ingénieur Statisticien Economiste en 2ème année*  
        - **Yague DIOP**,   *Elève Ingénieur Statisticien Economiste en 2ème année* 
        
        Ecole nationale de la Statistique et de l'Analyse Economique PIERRE NDIAYE
        
        **Superviseur** :
        
        **Mously DIAW**, *Data scientist*
        """)
    
    
# Page pour la présentation des données
def presentation_page(df):
    # Titre de la page
    st.title("Autour de la base de données")
    st.write(""" 
             Dans cette section, nous présentons un récapitulatif général des caractéristiques de la base de données. 
             """)
    # Boites pour afficher le nombre de lignes, de colonnes, de doublons
    n_variables = df.shape[1]
    n_observations = df.shape[0]
    n_duplicates = df.duplicated().sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Nombre de variables", n_variables)
    with col2:
        st.metric("Nombre d'observations", n_observations)
    with col3:
        st.metric("Nombre de doublons", n_duplicates)

    # Affichage du jeu de données et utilisation d'un slider pour incrémenter les lignes à afficher.
    st.subheader("Aperçu des données")
    n_rows = st.slider("Nombre de lignes à afficher", min_value=5, max_value=min(50, n_observations), value=5)
    st.dataframe(df.head(n_rows), use_container_width=True)

    # Tableau pour donner une information sur les variables ainsi que sur leur type. 
    st.subheader("Nature des variables")
    variables_info = pd.DataFrame({
        "Nom": df.columns,
        "Type": [df[col].dtype for col in df.columns],
        "Non-NA Count": df.notna().sum()
    })
    st.dataframe(variables_info, use_container_width=True)

    # Analyses descriptives des variables quantitatives
    st.subheader("Analyse descriptive (variables Quantitatives)")
    desc_stats = df.describe().transpose()
    st.dataframe(desc_stats, use_container_width=True)


def visualization_page(df):
    st.title("Visualisation des Données")
    st.markdown("Cette section présente les analyses sous forme de graphes de toutes les variables: univariées, bivariées, corrélation et nuage de points")
    st.subheader("Graphiques univariés")
    # Variable à selectionner
    variable = st.selectbox("Sélectionner une variable à visualiser", df.columns)
    # Définition des palettes de couleurs avec plotly
    valid_palettes = ["Plotly", "Set1", "Set2", "Set3", "Pastel1", "Dark2"]
    # Définition du graphe pour les variables quantitatives
    if df[variable].dtype in ['float64', 'int64']:
        # Quantitative variable options
        plot_type = st.radio("Choisir le type de graphique", ["Box-Plot", "Histogramme"])
        color = st.color_picker("Choisir une couleur pour le graphique", "#00FFAA")
        # Permettre à l'utilisateur de renommer les axes
        x_axis_title = st.text_input("Nom de l'axe X", "")
        y_axis_title = st.text_input("Nom de l'axe Y", variable)
        graph_title = st.text_input("Titre du graphique", plot_type)
        # Box-plot
        if plot_type == "Box-Plot":
            fig = px.box(df, y=variable, color_discrete_sequence=[color])
        elif plot_type == "Histogramme":
            nbins = st.slider("Nombre de bins", min_value=5, max_value=100, value=30)
            bar_gap = st.slider("Espacement entre les barres", min_value=0.0, max_value=0.5, value=0.1)
            fig = px.histogram(df, x=variable, nbins=nbins, color_discrete_sequence=[color])
            fig.update_traces(marker_line_width=bar_gap)
        # Mise a jour du titre des axes.
        fig.update_layout(
            title=graph_title,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title
        )
        # Affichage du graphes
        st.plotly_chart(fig, use_container_width=True)

        # Graphe si la variable est qualitative 
    elif df[variable].dtype == 'object':
        # Options de choix du type de graphe 
        plot_type = st.radio("Choisir le type de graphique", ["Camembert", "Barplot", "Anneau", "Entonnoir"])
        value_counts = df[variable].value_counts(normalize=True).reset_index()
        value_counts.columns = ["Catégorie", "Pourcentage"]
        palette = st.selectbox("Choisir une palette de couleurs", valid_palettes)
        #palette = st.text_input("Palette de couleurs (ex: Viridis, Plotly, etc.)", ["Plotly", "Viridis"])
        #palette = st.radio("Palette de couleurs (ex: Viridis, Plotly, etc.)", ["Plotly", "Viridis"])
        x_axis_title = st.text_input("Nom de l'axe X", "Catégorie")
        y_axis_title = st.text_input("Nom de l'axe Y", "Pourcentage")
        graph_title = st.text_input("Titre du graphique", plot_type)
        # Application du graphe selon que l'utilisateur choisisse le type de graphe : camembert, Anneu, entonnoir, barplot
        if plot_type == "Camembert":
            fig = px.pie(value_counts, names="Catégorie", values="Pourcentage", hole=0, color_discrete_sequence=getattr(px.colors.qualitative, palette))
        elif plot_type == "Barplot":
            fig = px.bar(value_counts, x="Catégorie", y="Pourcentage", text="Pourcentage", color="Catégorie",
                         color_discrete_sequence=getattr(px.colors.qualitative, palette))
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        elif plot_type == "Anneau":
            fig = px.pie(value_counts, names="Catégorie", values="Pourcentage", hole=0.5, color_discrete_sequence=getattr(px.colors.qualitative, palette))
        elif plot_type == "Entonnoir":
            fig = px.funnel(value_counts, x="Pourcentage", y="Catégorie", color="Catégorie",
                            color_discrete_sequence=getattr(px.colors.qualitative, palette))

        # Formatage des axes
        fig.update_layout(
            title=graph_title,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title
        )
        # Affichage des graphes 
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap des corrélations
    st.subheader("Matrice de Corrélations")
    # Selection des variables qui sont uniquement quantitatives 
    quant_vars = [col for col in df.columns if df[col].dtype in ["float64", "int64"]]
    selected_quant_vars = st.multiselect("Sélectionner les variables pour la heatmap :", quant_vars)

    # Matrice de corrélation
    if selected_quant_vars:
        # Calcul de la matrice de corrélation selon la formule de spearman
        corr_matrix = df[selected_quant_vars].corr(method="spearman")
        # Suppression des diagonales
        mask = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
        # Création du heatmap
        fig = go.Figure()
        # Ajout des valeurs de corrélatio,
        for i, row in enumerate(mask.index):
            for j, col in enumerate(mask.columns):
                if pd.notna(mask.iloc[i, j]):
                    fig.add_trace(go.Scatter(
                        x=[col],
                        y=[row],
                        text=[f"{mask.iloc[i, j]:.2f}"],
                        mode="text",
                        textfont=dict(color="black", size=12)
                    ))
        # Formatage
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="Viridis",
            zmin=-1,
            zmax=1,
            showscale=True
        ))
        # Titres des axes
        fig.update_layout(
            title="Heatmap des Corrélations",
            xaxis=dict(title="Variables"),
            yaxis=dict(title="Variables")
        )
        # Affichage du graphe
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Message d'erreur à afficher
        st.warning("Veuillez sélectionner au moins deux variables quantitatives.")


    # Nuage de points pour les variables quantitatives
    st.subheader("Nuage de Points")
    # Récupération des variables quantitatives
    quant_vars = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    cat_vars = [col for col in df.columns if df[col].dtype == 'object']
    # Condition sur le nombre de variables et réalisation du nuage de pointd 
    if len(quant_vars) >= 2:
        x_var = st.selectbox("Variable X", quant_vars)
        y_var = st.selectbox("Variable Y", quant_vars)
        color_var = st.selectbox("Variable de Couleur (Optionnel)", [None] + cat_vars)
        # Affichage du graphe nuage de points
        fig = px.scatter(df, x=x_var, y=y_var, color=color_var)
        # Gestion des titres des axes
        x_axis_title = st.text_input("Nom de l'axe X pour le nuage de points", x_var)
        y_axis_title = st.text_input("Nom de l'axe Y pour le nuage de points", y_var)
        graph_title = st.text_input("Titre du nuage de points", "Nuage de Points")
        # Mise à jour une fois les titres écris
        fig.update_layout(
            title=graph_title,
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title
        )
        # Affichage du graphe
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Message d'erreur au cas où les variables ne s'affichent pas. 
        st.warning("Pas assez de variables quantitatives pour un nuage de points.")
        

# Modèles avec hyperparamètres de base pour l'optimisation
MODELS_WITH_PARAMS = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge Regression": (Ridge(), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
    "Lasso Regression": (Lasso(), {"model__alpha": [0.01, 0.1, 1.0, 10.0]}),
    "Elastic Net": (ElasticNet(), {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}),
    "Random Forest": (RandomForestRegressor(), {"model__n_estimators": [100, 200], "model__max_depth": [5, 10]}),
    "Gradient Boosting": (GradientBoostingRegressor(), {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1]}),
}

# Page de la modélisation
def prediction_page(df):
    st.title("Modélisation")
    st.markdown("Cette section permet de prédire à partir du modèle adéquat la performance des étudiants.")
    # Sélection des variables
    target = st.selectbox("Sélectionner la variable cible", df.select_dtypes(include=[np.number]).columns)
    features = st.multiselect("Sélectionnez les variables explicatives :", df.columns)
    test_size = st.slider("Pourcentage des données de test :", 10, 50, 20) / 100

    # Initialiser les variables catégorielles et numériques en amont
    num_vars = df.select_dtypes(exclude="object").columns
    cat_vars = df.select_dtypes(include="object").columns

    # Vérifiez si les modèles ont déjà été entraînés dans `st.session_state`
    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False  # État initial

    # Étape d'entraînement des modèles
    if st.button("Exécuter les modèles"):
        if not features:
            st.error("Veuillez sélectionner des variables explicatives.")
            return

        X = df[features]
        y = df[target]

        # Mise à jour des variables catégorielles et numériques spécifiques aux features
        cat_vars = X.select_dtypes(include="object").columns
        num_vars = X.select_dtypes(exclude="object").columns

        # Mise en place du préprocesseur avec imputation
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_vars),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_vars),
            ]
        )

        # Split des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        results = []
        pipelines = {}

        # Modélisation avec tous les modèles
        for name, (model, params) in MODELS_WITH_PARAMS.items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)
            results.append({"Model": name, "MSE": mse, "R2": r2, "Adj R2": adj_r2})
            pipelines[name] = pipeline

        results_df = pd.DataFrame(results).sort_values(by=["R2", "MSE", "Adj R2"], ascending=[False, True, False])
        st.markdown("Résultats des modèles :")
        st.table(results_df)

        # Sauvegarder les résultats dans `st.session_state`
        st.session_state.models_trained = True
        st.session_state.results_df = results_df
        st.session_state.pipelines = pipelines
        st.session_state.cat_vars = cat_vars
        st.session_state.num_vars = num_vars

    # Si les modèles ont été entraînés, afficher les résultats
    if st.session_state.models_trained:
        results_df = st.session_state.results_df
        pipelines = st.session_state.pipelines
        cat_vars = st.session_state.cat_vars  # Récupérer les variables catégorielles
        num_vars = st.session_state.num_vars  # Récupérer les variables numériques

        # Sélection du meilleur modèle
        best_model_name = results_df.iloc[0]["Model"]
        best_model_pipeline = pipelines[best_model_name]
        st.write(f"Meilleur modèle : {best_model_name}")

        # Ajout de la rubrique "Prédiction"
        st.markdown("### Prédiction")
        input_data = {}

        # Création de 4 colonnes pour afficher les champs d'entrée
        cols = st.columns(4)  # 4 colonnes pour afficher les champs sous forme de grille
        
        col_idx = 0  # Index de la colonne à utiliser

        for col in features:
            if col in num_vars:
                # Pour les variables numériques, saisir une valeur quantitative positive
                with cols[col_idx]:
                    input_data[col] = st.number_input(f"Entrez la valeur pour {col}", min_value=0.0, key=f"input_{col}")
            elif col in cat_vars:
                # Pour les variables catégorielles, afficher un choix parmi les modalités uniques
                with cols[col_idx]:
                    categories = df[col].dropna().unique()
                    input_data[col] = st.selectbox(f"Choisissez la catégorie pour {col}", categories, key=f"select_{col}")

            # Passer à la colonne suivante après chaque champ
            col_idx += 1
            if col_idx == 4:
                col_idx = 0  # Revenir à la première colonne après 4

        # Convertir les données d'entrée en DataFrame
        input_df = pd.DataFrame([input_data])

        # Prédiction avec le modèle sélectionné
        if st.button("Faire la prédiction"):
            predicted_value = best_model_pipeline.predict(input_df)
            st.write(f"La prédiction pour la variable cible est : {predicted_value[0]:.2f}")

if __name__ == "__main__":
    main()
# FIN
