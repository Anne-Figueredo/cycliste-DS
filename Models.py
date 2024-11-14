import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error

# Charger les données ML
@st.cache_data
def load_ml_data():
    df_ml = pd.read_csv('comptage-velo-ML-sans-sc.csv', sep=',')
    df_ml = df_ml.set_index(['date_comptage', 'nom_du_site_de_comptage'])
    return df_ml

# Charger les données ML
df_ml = load_ml_data()


# Liste des modèles à tester avec leurs hyperparamètres prédéfinis
models = {
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet()
}

# Liste des hyperparamètres GRIDSEARCH pour chaque modèle
best_params_gs = {
    'Random Forest Regressor': {'max_depth': None, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50},
    'Gradient Boosting': {'learning_rate': 0.1, 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50},
    'Lasso': {'alpha': 0.01},
    'Ridge': {'alpha': 1.0},
    'ElasticNet': {'alpha': 0.01, 'l1_ratio': 0.5}
}

# Liste des hyperparamètres RANDOMIZEDSEARCH pour chaque modèle
best_params_rs = {
    'Random Forest Regressor': {'max_depth': 15, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 135},
    'Gradient Boosting': {'learning_rate': 0.2, 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 58},
    'Lasso': {'alpha': 0.01},
    'Ridge': {'alpha': 1.0},
    'ElasticNet': {'l1_ratio': 0.5, 'alpha': 0.01}
}

def train_and_save_models(models, df_ml, best_params_gs, best_params_rs):
    mean_results = {} 
    results_compteur = {}
    predicts_compteur_df = pd.DataFrame(index=df_ml.index)  # DataFrame pour stocker les prédictions
    predictions_list = []  # Initialisation de la liste de prédictions

    # Appliquer le StandardScaler sur tout le DataFrame
    sc = StandardScaler()
    sc.fit(df_ml)
    df_ml_scaled = pd.DataFrame(sc.transform(df_ml), columns=df_ml.columns, index=df_ml.index)
    
    for model_name, model in models.items():
        for param_type in ['Prédéfinis', 'Best Params GridSearch', 'Best Params RandomizedSearch']:
            hyperparams = None
            if param_type == 'Best Params GridSearch':
                hyperparams = best_params_gs.get(model_name)
            elif param_type == 'Best Params RandomizedSearch':
                hyperparams = best_params_rs.get(model_name)

            if hyperparams is not None:
                model.set_params(**hyperparams)

            train_scores = []
            test_scores = []
            maes = []
            predictions_dict = {}
            all_predictions = pd.DataFrame()

            for compteur_id in df_ml_scaled.index.get_level_values(1).unique():
                #filtrer le compteur_data de la boucle à travailler
                compteur_data = df_ml_scaled.loc[(slice(None), compteur_id), :]
                #séparer en deux 
                features = compteur_data.drop(['comptage_horaire'], axis=1)
                target = compteur_data['comptage_horaire']
                #appliquer le traintestsplit
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

                model.fit(X_train, y_train)
                #calculer prédictions
                y_pred = model.predict(X_test)
                
                # Calculer le MAE
                mae = mean_absolute_error(y_test, y_pred)

                #enregistrer MAE 
                maes.append(mae)

                #calculer les scores et enregistrer
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                train_scores.append(train_score)
                test_scores.append(test_score)

                 # Ajouter les prédictions à all_predictions
                predictions_df = pd.DataFrame({'Prédictions': y_pred}, index=X_test.index)
                all_predictions = pd.concat([all_predictions, predictions_df])

                # Enregistrer les scores par compteur dans results_compteur
                results_compteur[f"{model_name}_{param_type}_{compteur_id}"] = {
                    'Train Score': train_score,
                    'Test Score': test_score,
                    'MAE': mae
                }
            
            # Enregistrer toutes les prédictions dans predictions_dict
            key = f"{model_name}_{param_type}"
            predictions_dict[key] = all_predictions

            # Enregistrer le dictionnaire dans un fichier Joblib
            joblib.dump(predictions_dict, f"predictions_{model_name}_{param_type}.joblib")

                





   
    #Boucle pour les moyennes des compteurs
    for model_name, model in models.items():
        mean_results[model_name] = {}
        for param_type in ['Prédéfinis', 'Best Params GridSearch', 'Best Params RandomizedSearch']:
            mean_results[model_name][param_type] = {
                'Mean Train Score': sum(train_scores) / len(train_scores),
                'Mean Test Score': sum(test_scores) / len(test_scores),
                'Mean MAE': sum(maes) / len(maes)
            }



    # Enregistrement des résultats et les prédictions dans des fichiers Joblib
    joblib.dump(mean_results, "model_mean_results.joblib")
    joblib.dump(results_compteur, 'model_results_compteur.joblib')
    joblib.dump(predicts_compteur_df, 'predicts_compteur_df.joblib')

# Appel de la fonction pour entraîner les modèles et enregistrer les résultats
train_and_save_models(models, df_ml, best_params_gs, best_params_rs)

# Afficher le message de confirmation
st.write("Les résultats des modèles et les prédictions ont été enregistrés avec succès.")

# Visualiser les enregistrements
st.write('Les moyennes des résultats : ')
mean_results_ok = joblib.load("model_mean_results.joblib")
st.write(mean_results_ok)

st.write('Les résultats par compteurs : ')
results_compteur = joblib.load("model_results_compteur.joblib")
df_results_compteur = pd.DataFrame(results_compteur)
df_results_compteur = df_results_compteur.transpose()
st.dataframe(df_results_compteur)

predictions_Randomh = joblib.load("predictions_Random Forest Regressor_Best Params GridSearch.joblib")
predictions_df = pd.concat(predictions_Randomh.values())
st.dataframe(predictions_df)
for level in predictions_df.index.levels:
    st.write(f"Valeurs uniques pour le niveau '{level.name}':")
    st.write(predictions_df.index.get_level_values(level.name).unique())

