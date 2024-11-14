import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import folium
import geopandas as gpd
import joblib
import contextily as ctx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

#mise en cache du df pour améliorer la performance
@st.cache_data
def load_old_data():
    df_old = pd.read_csv('comptage-velo-donnees-compteurs.csv', sep=';')
    return df_old

@st.cache_data
def load_data_clean():
    df = pd.read_csv('comptage-velo-CLEAN.csv', sep=',')
    return df

df_old = load_old_data()
df = load_data_clean()

st.sidebar.image("velo-gif2.gif", width= None)
st.sidebar.title("Sommaire")



pages=["Présentation du projet", "Exploration des données", "DataVizualization", "Cartographie","Clustering", "Modélisation", "Conclusions"]
page=st.sidebar.radio("Aller vers", pages)


#PAGE 1 : PRESENTATION DU PROJET
if page == pages[0]:
  st.title("Comment analyser et prédire le trafic cycliste Parisien")
  st.write("Par Anne Levet-Figueredo, Karine Mhamdi, Kaity Kone")
  st.write("Cohorte DA septembre 2023")
  st.image('bis.jpg')
  st.write("### Présentation du projet")
  st.write("A part pour Karine qui a spécifiquement sélectionné ce thème, ce projet a été attribué à Anne et Kaity. Il ne concerne pas nos entreprises actuelles et nous nous sommes donc imprégnées des données et du contexte chacune en même temps.")
  st.write("Dans un contexte d'environnement climatique tendu, nous sommes tous sensibilisés à l’enjeu des moyens de transport éco responsable depuis plusieurs années, dont le vélo est largement préconisé.")
  st.write("En effet, d’après le site du gouvernement.fr, l’utilisation du vélo est mise en avant grâce à 6 principales raisons :")
  st.write("- réduire de 30% le risque de maladie")
  st.write("- moins de risque d’être blessés qu’en voiture")
  st.write("- économiser 650 kilogrammes de CO2 par personne/an")
  st.write("- c’est le moyen de déplacement le plus performant pour les trajets de moins de 5km")
  st.write("- le coût des dépenses est de 100€/an à vélo contre 1 000€/an en voiture")
  st.write("- plus de problèmes de stationnement")

  st.write("Pour des trajets travail/maison ainsi que les balades dominicales, surtout à Paris, 42ème ville la plus polluée d’après https://www.iqair.com/fr/world-air-quality-ranking, le vélo est un moyen de transport majeur.") 

  st.write("Plusieurs primes à destination des employeurs et des particuliers sont d’ailleurs mises en place afin de pousser à l’utilisation des mobilités douces et durables comme le FMD.")

  st.write("A partir du dataset de bornes de comptage de passages de vélos qui nous a été fourni, nous réalisons une étude qui consiste à identifier les corrélations entre la variable cible de fréquentation et les autres variables catégorielles et explicatives. Nous souhaitons principalement mettre en avant l’utilisation des bornes Velibs à Paris afin de faciliter leur travaux d’amélioration (augmentation du nombre de velibs disponibles aux stations) en fonction des prédictions que nous effectuerons grâce au Machine Learning sur les 7/8/9/10 prochains jours en temps réel.")

  st.write("L’objectif de cette étude est de prédire la fréquentation cycliste afin de faciliter la mobilité dans la ville de Paris grâce à l’amélioration de deux axes : amélioration des plans des pistes cyclables en fonction des lieux culturels et événementiels à Paris prédictions de la quantité de vélos disponibles aux différentes stations de velibs à J+1. Le Dataset étant sur une année réelle de octobre 2022 à novembre 2023, nous pourrons aussi envisager de prédire les fréquentations sur les zones des stations de velibs. Cette étude permettra de visualiser si les capacités des bornes de velibs sont cohérentes en fonction du nombre de passages de la zone afin d’envisager des travaux de réaménagement..")

  st.write("Nous notons que les compteurs de passage horaire quantifient le passage de tous types de velos : usagers, velibs, electriques… dans les sens aller et retour dont nous utiliserons le cumul.")



#PAGE 2 : PRESENTATION DES DATASETS
if page == pages[1] : 
  st.title("Présentation des datasets")

  tab1, tab2 = st.tabs(["Dataset initial : comptage-velos-donnees-compteurs", "Dataset complémentaires"])
  with tab1:
    st.header('1. Source')
    st.write('Nous avons un jeu de données des comptages horaires de vélos par compteur et de localisation des sites de comptage en J-1 sur 13 mois glissants du 01/10/2022 au 12/11/2023.')
    st.header('2. Dimension')
    st.write('Voici une visualisation du dataset nommé df :')
    st.dataframe(df_old.head(10))
    st.write('La dimension du dataset est :', df_old.shape)
    types_df_old = pd.DataFrame(df_old.dtypes, columns=['Type de données'])
    st.write('Information du df :') 
    st.dataframe(types_df_old, width=300)  
    st.write('Description du df :')
    st.dataframe(df_old.describe())
    
  with tab2:
    st.header('1. Source')
    st.write('Meteo : https://donneespubliques.meteofrance.fr/')
    st.write('Évènement : https://opendata.paris.fr/')
    st.write('Jours feriés : https://www.data.gouv.fr/fr/datasets/jours-feries-en-france/')
    st.write('Vacances : https://www.data.gouv.fr/fr/datasets/le-calendrier-scolaire/')
    st.header('2. Dimensions')
    st.write('XXXXXXXXXXXXXXXXXXXXX')
    st.dataframe(df.head(10))
    st.write('XXXXXXXXXXXXXXXXXXX')
    st.write('### Information du df :')
    st.dataframe('XXXXXXXXXXXXXXXXXXXX')
    st.write('### Description du df :')
    st.dataframe('XXXXXXXXXXXXXXXXXXXXX')

  

#PAGE 3 DATAVISUALIZATION
if page == pages[2] : 
  st.title("DataVizualization")
 
  tab1, tab2, tab3 = st.tabs(["Performances", "Temporelle", "Météo"])


  with tab1:
    st.header('Performance des sites des bornes')
    #TOP 10 DES SITES LES PLUS FREQUENTES
    st.subheader('TOP 10 des sites ayant le plus de fréquentation')
    max_values = df.groupby('nom_du_site_de_comptage')['comptage_horaire'].sum().reset_index().nlargest(10, 'comptage_horaire')
    hist = px.bar(max_values, x='nom_du_site_de_comptage',y='comptage_horaire')
    st.plotly_chart(hist)

    st.divider()

    #TOP 10 DES SITES LES PLUS FREQUENTES
    st.subheader('FLOP 10 des sites ayant le moins de fréquentation')
    min_values = df.groupby('nom_du_site_de_comptage')['comptage_horaire'].sum().nsmallest(10).reset_index()
    hist = px.bar(min_values, x='nom_du_site_de_comptage', y='comptage_horaire')
    st.plotly_chart(hist)


    #FREQUENTATION DYNAMIQUE PAR BORNES AU CHOIX
    st.subheader('Fréquentation des bornes par mois par sens de passage')
    sites_de_comptage = df['nom_du_site_de_comptage'].unique()
    site_selectionne = st.selectbox('Choisissez un site de comptage :', sites_de_comptage)
    # Filtrer les données pour n'inclure que les données du site de comptage sélectionné
    donnees_site_selectionne = df.loc[df['nom_du_site_de_comptage'] == site_selectionne]

    fig = px.bar(donnees_site_selectionne, x='mois_annee_comptage', y='comptage_horaire', color='nom_du_compteur',
         title=f'Fréquentation aller-retour par bornes pour le site {site_selectionne}',
         labels={'mois_annee_comptage': 'Mois et année', 'comptage_horaire': 'Comptage horaire'}, barmode='group')

    graphique_section = st.empty()
    graphique_section.plotly_chart(fig)

    
  with tab2:
    st.header('Analyse temporelle')
     
    #MISE EN CACHE DU DATA FILTRE MAX VALUES
    @st.cache_data
    def filtered_df(df):
      max_values = df.groupby('nom_du_site_de_comptage')['comptage_horaire'].sum().reset_index().nlargest(10, 'comptage_horaire')
      return df[df['nom_du_site_de_comptage'].isin(max_values['nom_du_site_de_comptage'])]

    df_filtered = filtered_df(df)

    #LINEPLOT EVOLUTION PAR MOIS
    st.subheader('Evolution du comptage par mois sur les 10 meilleurs sites')
    plt.figure(figsize=(15, 7))
    sns.lineplot(x='mois_annee_comptage', y='comptage_horaire', hue='nom_du_site_de_comptage', data=df_filtered)
    plt.title('Top 10 des sites : évolution par mois')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    st.pyplot(plt)
    st.write("L'évolution est quasi-constante de janvier à mai. Puis une augmentation en juin (beaux jours), une baisse en juillet et aout (vacances scolaires) et une grosse reprise en septembre (rentrée scolaire). Le traffic décroit alors jusqu'à décembre (chute des températures).")

    st.divider()

    #LINEPLOT EVOLUTION PAR HEURE
    st.subheader('Evolution du comptage par heure sur les 10 meilleurs sites')
    plt.figure(figsize=(15, 7))
    sns.lineplot(x='heure_comptage', y='comptage_horaire', hue='nom_du_site_de_comptage', data=df_filtered)
    plt.title('Top 10 des sites : évolution par heure')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    st.pyplot(plt)
    st.write("On note deux pics d'activités vers 8h et 17h, ce qui correspond aux trajets de travail.")
    
    st.divider()

    #MISE EN CACHE DU DATA FILTRE MOYENNE
    @st.cache_data
    def group_data(df):
      return df.groupby(['mois_annee_comptage'])['comptage_horaire'].mean().reset_index()
    grouped_data = group_data(df)

    #BARPLOT MOYENNE DES PASSAGES
    st.subheader('Moyenne des passages par mois sur tous les compteurs')
    plt.figure(figsize=(15, 7))
    sns.barplot(x='mois_annee_comptage', y='comptage_horaire', data=grouped_data)
    plt.title('Moyenne des passages par mois')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    st.write("L'évolution de la moyenne de tous les compteurs correspond à l'évolution des 10 meilleurs sites vu plus haut.")

    st.divider()

    #COMPARASON SEMAINE / WEEKEND
    st.subheader('Comparaison semaine et week-end')
    st.write("En deux jours de weekend, il y a plus de la moitié de fréquentation que sur 5 jours de semaine.")
    plt.figure(figsize=(15, 7))
    sns.barplot(x='week_end',y='comptage_horaire', data=df)
    plt.title("Comptage par semaine")
    plt.xticks([0, 1], ['Jour de semaine', 'Week-end'])
    st.pyplot(plt)

    col1, col2 = st.columns(2)

    with col1:
      #MISE EN CACHE DU DATA FILTRE TOP 10 SEMAINE
      @st.cache_data
      def top_10_semaine(df):
       return df[(df['nom_du_site_de_comptage'].isin(max_values['nom_du_site_de_comptage'])) & (df['week_end'] == 0)]
      top_10_semaine = top_10_semaine(df)

      #LINEPLOT EVOLUTION HEURES SEMAINES
      st.write('Evolution du comptage de passage par heure à la semaine')
      plt.figure(figsize=(15, 6))
      sns.lineplot(x='heure_comptage', y='comptage_horaire', hue='nom_du_site_de_comptage', data=top_10_semaine)
      plt.title('Top 10 des sites : évolution par heure à la semaine')
      plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
      st.pyplot(plt)

    with col2:
      #MISE EN CACHE DU DATA FILTRE TOP 10 WEEKEND
      @st.cache_data
      def top_10_weekend(df):
        return df[(df['nom_du_site_de_comptage'].isin(max_values['nom_du_site_de_comptage'])) & (df['week_end'] == 1)]
      top_10_weekend = top_10_weekend(df)

      #LINEPLOT EVOLUTION HEURES WEEKEND
      st.write('Evolution du comptage de passage par heure le week-end')
      plt.figure(figsize=(15, 6))
      sns.lineplot(x='heure_comptage', y='comptage_horaire', hue='nom_du_site_de_comptage', data=top_10_weekend)
      plt.title('Top 10 des sites : évolution par heure le weekend')
      plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
      st.pyplot(plt)

    st.write("Les pics d'heures de fréquentation de la semaine correspondent bien à des trajets maison-travail. Le weekend l'évolution est totalement différente. L'activité loisir est présente toute au long de la journée de 5h à 20h.")

    st.divider()

    #BARPLOT PAR VACANCES ET FERIES
    st.subheader('Comparaison des passages avec les vacances et feriés')
    plt.figure(figsize=(15, 6))
    sns.barplot(x='vacances',y='comptage_horaire', data=df)
    plt.title("Comptage des passages vacances")
    st.pyplot(plt)
    st.write("Même si la fréquentation hors vacances est supérieure, la fréquentation pendant la vacances est conséquente. Ceci s'explique aussi du fait de la dimension touristique de la ville qui attire beaucoup de touristes et de fréquentation pedestre et cycliste.")




  with tab3:
    st.header('Météo')

    #BARPLOT PAR METEO
    st.subheader('Evolution du passage par conditions météorologiques')
    plt.figure(figsize=(15, 6))
    sns.barplot(x='icon', y='comptage_horaire', data=df)
    plt.title('Comptage des passages par conditions météorologiques')
    st.pyplot(plt)
    st.write("Les meilleurs conditions météorologiques sont 'ensoleillé' et 'partiellement nuageux'. Sans surprise, les temps 'neigeux' et 'pluie' sont moins prisés pour la pratique du vélo.")

    st.divider()

    #BARPLOT PAR SAISON
    st.subheader('Evolution du passage par saison du 01/10/22 au 30/09/23')
    plt.figure(figsize=(15, 6))
    sns.barplot(x='saison', y='comptage_horaire', data=df[(df['date_comptage'] >= '2022-10-01') & (df['date_comptage'] <= '2023-09-30')])
    plt.title("Comptage des passages par saison")
    st.pyplot(plt)
    st.write("Avec les chutes de températures et les conditions météos moins propices (pluie, neige, vent), l'hiver est la saison qui compte le moins de fréquentation cycliste.")


#PAGE 4 VELIBS ET PISTES CYCLABLES
if page == pages[3] : #Cartographie
  st.title("Cartographie")
  tab1, tab2, tab3, tab4 = st.tabs(["Piste Cyclable", " Velib' autour de 500 m ", "Velib' groupés autour de 500m", "Sites Touristiques"])


  with tab1:
  #Carte pistes cyclables_compteurs
    st.image("piste_cycl&compteurs.png", width= None)
    st.write("On peut constater que les sites les plus représentatifs en termes de comptages se situent sur les pistes qui mènent aux sites touristiques emblématiques de Paris comme l’Arc de Triomphe, le Trocadéro et la Tour Eiffel mais également sur les pistes passant par le Grand Palais, l'Elysée ou celles menant aux Invalides et au quartier de Montparnasse")
    

    
    @st.cache_data
    def load_velibs():
      velibs = gpd.read_file('velib-emplacement-des-stations.geojson', sep=';')
      return velibs

    velibs = load_velibs()

    map = velibs.explore()
    from streamlit_folium import folium_static
    folium_static(map)
    st.write("On note que la ville de Paris est bien équipée en stations velib, tout arrondissement confondu, et même en banlieue proche")
    st.markdown('<hr style="border:2px solid #e1e1e1"> </br>', unsafe_allow_html=True)

  with tab2:
    st.write("L'objectif est de déterminer combien de stations Vélib' se trouvent dans un rayon de 500 mètres autour de chaque site de comptage, ce qui permettrait d'évaluer l'accessibilité des vélos. Cette approche nous permettra de mieux comprendre la couverture actuelle du réseau Vélib' par rapport aux sites de comptage.")
    st.image("buffer500m_compteurs.png", width= None)
  
  with tab3: 
    st.write("Pour faciliter l'analyse et rendre les résultats plus lisibles, nous allons regrouper les buffers qui se chevauchent ou qui sont proches les uns des autres.")
    st.write("Nous nous rendons compte  que seulement 10% des stations Vélib' sont situées dans un rayon de 500 mètres des sites de comptage de vélos. Cela indique une répartition inégale des stations Vélib' par rapport aux points de comptage")
    st.image("buffer_regrouper_compteurs.png", width= None)
    
  with tab4:
    st.write("La plupart des sites touristiques sont à moins de 1 km des sites de comptage, mais en dehors du rayon de 500 m. Ils sont accessibles en vélo grâce à la présence de nombreuses stations Vélib' à proximité")
    st.image("touristique_velib.png", width= None)
   


  

#PAGE 5 CLUSTERING
if page == pages[4] : 
  st.title("Clustering")
  st.write(" Nous avons choisi d'effectuer une analyse des données grâce à la technique du clustering afin d'identifier les schémas de comportement des cyclistes dans différentes zones de Paris. En regroupant les sites de comptage en fonction de leur activité horaire, nous pourrions segmenter les utilisateurs de vélos en différents groupes en fonction de leurs habitudes de déplacement. Cela peut être utile pour adapter les politiques de transport ou pour les entreprises de partage de vélos pour optimiser leurs services.")
 
  st.write("Le choix du nombre de clusters à former a été déterminé par la méthode du coude, elle permet d’identifier visuellement le nombre de clusters le plus pertinent.")
  st.image("Coude.png", width= None)
  st.write("Dans ce cas, nous avons choisi de déterminer un nombre de clusters optimal à 5.")


  st.write("L’algorithme de clustering K-means a été appliqué aux données de comptage horaire pour former des groupes homogènes en fonction des tendances horaires. Chaque cluster représente un modèle de trafic spécifique observé à différentes heures de la journée.")
  st.image("pistes_cluster.png", width= None)
  st.write("On constate des activités horaires similaires localisées au Sud Ouest de Paris, représentées par le cluster de couleur mauve ou encore à l'Est de Paris représentées par le cluster de couleur verte")






#PAGE 6 MODELISATION
if page == pages[5] : 
  st.title("Modélisation")


  #MISE EN CACHE DU DATA ML
  @st.cache_data
  def load_ml_data():
      df_ml = pd.read_csv('comptage-velo-ML-sans-sc.csv', sep=',')
      df_ml = df_ml.set_index(['date_comptage', 'nom_du_site_de_comptage'])
      return df_ml
  df_ml = load_ml_data()


  #Création des onglets
  tab1, tab2, tab3 = st.tabs(["Simulation Machine Learning", "Préparation des données", "Comparaison des résultats des modèles"])

  with tab1:
    st.header('Simulation de Machine Learning')

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
        'Lasso' : {'alpha': 0.01},
        'Ridge' : {'alpha': 1.0},
        'ElasticNet' : {'alpha': 0.01, 'l1_ratio': 0.5}
    }

    # Liste des hyperparamètres RANDOMIZEDSEARCH pour chaque modèle
    best_params_rs = {
        'Random Forest Regressor': {'max_depth': 15, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 135},
        'Gradient Boosting': {'learning_rate': 0.2, 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 58},
        'Lasso' : {'alpha': 0.01},
        'Ridge' : {'alpha': 1.0},
        'ElasticNet' : {'l1_ratio': 0.5, 'alpha': 0.01}
    }

    model = None

    # Interface utilisateur pour la sélection des hyperparamètres
    selected_compteur = st.selectbox('Sélectionnez le compteur :', df_ml.index.get_level_values(1).unique())

    # Interface utilisateur pour la sélection du modèle
    selected_model = st.selectbox('Sélectionnez le modèle à simuler :', list(models.keys()))

    # Interface utilisateur pour la sélection des hyperparamètres
    selected_hyperparams = st.selectbox('Sélectionnez les hyperparamètres :', ['Prédéfinis', 'Best Params GridSearch', 'Best Params RandomizedSearch'])


   # JOBLIB AVERAGE RESULTS DES COMPTEURS OK
    mean_results = joblib.load("model_results.joblib")

    #JOBLIB RESULTS PAR COMPTEURS OK
    results_compteur = joblib.load("model_results_compteur.joblib")
    df_results_compteur = pd.DataFrame(results_compteur)
    df_results_compteur = df_results_compteur.transpose()


    #JOBLIB PREDICTS PAR COMPTEURS NON
    # Charger les prédictions en fonction des combinaisons de model_name et param_type sélectionnées
    def load_predictions(model_name, param_type):
      predictions_dict = joblib.load(f"predictions_{model_name}_{param_type}_{compteur_id}.joblib")
      return predictions_dict




    model = models[selected_model]
    # Dictionnaire pour stocker les y_test correspondant à chaque compteur
    y_tests = {}


    #Standard Scaler SUR TOUT LE DF
    sc = StandardScaler()
    sc_all = sc.fit_transform(df_ml)
    df_ml = pd.DataFrame(sc_all, columns=df_ml.columns, index=df_ml.index)
    df_ml.head()



    for compteur_id in df_ml.index.get_level_values(1).unique():
      compteur_data = df_ml.loc[(slice(None), compteur_id), :]
      features = compteur_data.drop(['comptage_horaire'], axis=1)
      target = compteur_data['comptage_horaire']
      X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
      y_tests[compteur_id] = y_test

    radio = st.radio("Choisissez une option :", ["Lancer le modèle sur le jeu d'entrainement", "Lancer le modèle sur le data"])



    if radio == "Lancer le modèle sur le jeu d'entrainement":
      if results_compteur is not None:
        if st.button("Lancer le modèle sur le jeu d'entrainement"):
            #Sélectionne les résultats pour le modèle spécifié
            model_key = f"{selected_model}_{selected_hyperparams}_{selected_compteur}"  
            selected_results = df_results_compteur.loc[model_key]

            #Affiche les résultats
            st.write("Résultat du compteur sélectionné")
            st.dataframe(selected_results)

           # Utiliser les prédictions pour un compteur_id spécifique
            predict_key = f"predictions_{selected_model}_{selected_hyperparams}.joblib"
            predictions = joblib.load(predict_key)
            predictions_df = pd.concat(predictions.values())

          # Filtrer les prédictions en fonction de l'emplacement obtenu
            predictions_df_filtered = predictions_df.loc[predictions_df.index.get_level_values(1) == selected_compteur]
            predictions_df_filtered['Valeurs Réelles'] = y_tests[selected_compteur]
            st.dataframe(predictions_df_filtered)

          # Créer un graphique
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=predictions_df_filtered.index.get_level_values('date_comptage'), y='Prédictions', data=predictions_df_filtered, label='Prédictions')
            sns.lineplot(x=predictions_df_filtered.index.get_level_values('date_comptage'), y='Valeurs Réelles', data=predictions_df_filtered, label='Valeurs Réelles')
            plt.title('Comparaison des Prédictions du {} avec {} avec les Données Réelles'.format(selected_model, selected_hyperparams))
            plt.legend()
            st.pyplot(plt)




  



    if radio == "Lancer le modèle sur le data":
      # Bouton pour lancer le modèle avec les hyperparamètres sélectionnés
      if st.button('Lancer le modèle sur les données d\'origine'):
        compteur_data = df_ml.loc[(slice(None), selected_compteur), :]
        sc = StandardScaler()
        compteur_data_scaled = sc.fit_transform(compteur_data)
        compteur_data_scaled_df = pd.DataFrame(compteur_data_scaled, index=compteur_data.index, columns=compteur_data.columns)
        features = compteur_data_scaled_df.drop(['comptage_horaire'], axis=1)
        target = compteur_data_scaled_df['comptage_horaire']
    
        # Configuration des hyperparamètres
        hyperparams = None
        if selected_hyperparams == 'Best Params GridSearch':
            hyperparams = best_params_gs.get(selected_model)
        elif selected_hyperparams == 'Best Params RandomizedSearch':
            hyperparams = best_params_rs.get(selected_model)

        # Création du modèle
        model = models[selected_model]

        # Configuration du modèle avec les hyperparamètres choisis
        if hyperparams is not None:
            model.set_params(**hyperparams)

        # Entraînement du modèle
        model.fit(features, target)
       

        predictions = model.predict(features)
        mae = mean_absolute_error(target, predictions)
        score = model.score(features, target)

        st.write("Score du modèle sur les données d'origine :", score)
        st.write("Mean absolute error sur les données d'origine :", mae)

        # Créer un DataFrame pour afficher les prédictions
        st.write("Visualisation des valeurs")
        results = pd.DataFrame({'Prédictions': predictions, 'Vraies valeurs': target}, index=compteur_data.index)
        st.write(results.head(5))

        #Graphique
        plt.figure(figsize=(15, 6))
        df_compare = pd.DataFrame({'Date': features.index.get_level_values(0), 'Données Réelles': target, 'Prédictions': predictions})
        sns.lineplot(x='Date', y='value', hue='variable', data=pd.melt(df_compare, ['Date']))
        plt.title('Comparaison des Prédictions du {} avec {} avec les Données Réelles au compteur situé {}'.format(selected_model, selected_hyperparams, selected_compteur))
        plt.xlabel('Date de Comptage')
        plt.ylabel('Comptage Horaire')
        plt.legend(title='Type')
        st.pyplot(plt)

        # Prédire les valeurs avec le modèle entraîné
        predictions = model.predict(features)

        # Créer un DataFrame pour stocker les prédictions
        predictions_df = pd.DataFrame(predictions, index=features.index, columns=['Prédictions'])

        # Inverser la transformation du scaler sur les prédictions
        predictions_unscaled = sc.inverse_transform(predictions_df)

        # Créer un DataFrame pour les prédictions inversées
        predictions_unscaled_df = pd.DataFrame(predictions_unscaled, index=predictions_df.index, columns=predictions_df.columns)

        # Afficher les prédictions inversées
        st.write("Prédictions de J+7 après inversion du scaler :")
        st.write(predictions_unscaled_df)





  with tab2:
    st.write("Afin d'éviter la fuite de données et d'enrichir le jeu de données, nous avons créé de nouvelles variables :")
    st.write("- Les températures, les températures ressenties et les conditions météorologiques ont été supprimées après avoir été reportées dans de nouvelles variables pour J-7, J-8, J-9 et J-10.")
    st.write("- De même, le comptage horaire n'a pas été supprimé mais a été placé dans la variable cible (y), car il s'agit de notre objectif de prédiction.")



    types_df_ml = pd.DataFrame(df_ml.dtypes, columns=['Type de données'])
    st.dataframe(types_df_ml, width=800)

    # Afficher les données
    st.subheader('Données après One Hot Encoding et StandardScaler')

    st.dataframe(df_ml.head())

 






  with tab3:
    # Charger les résultats à partir du fichier Joblib
    results_compare = pd.read_csv('machinelearning-compare.csv', sep = ',', index_col = 0)

# Afficher le DataFrame

    st.write("Comparaison des moyennes résultats des modèles sur tous les compteurs")
    st.dataframe(results_compare)

    st.write("D'après ces résultats, le Random Forest sans hyperparamètres et le Gradient Boosting sans hyperparamètres et avec les paramètres du Grid Search sont assez similaires.")
    st.write("Il conviendrait de lancer ces modèles dans la Simulation de Machine Learning avec ces éléments. D'autres scores seraient à tester mais nous avons préféré nous arrêter à cette conclusion qui offre déjà de très bons résultats.")





#PAGE 7 CONCLUSIONS
if page == pages[6] : 
  st.title("Conclusions")

  st.write("Ces différents axes d'analyses nous permettent de retirer plusieurs informations :")

  st.write("- Les données sont cohérentes avec l'environnement Parisien et environnemental : cycle de travail-maison hors week-end, évolution du trafic en fonction des conditions météorologiques et des systèmes de vacances scolaires.")

  st.write("- Le clustering a mis en évidence des axes de fréquentation de passages en fonction des horaires journalières")

  st.write("- La cartographie nous en apprend beaucoup plus sur les localisations des bornes de compteurs par rapport aux pistes cyclables, aux stations de Velib's et aux lieux touristiques. Ainsi, nous avons pouvons proposer la création de nouvelles bornes de comptage autours de certains de ces lieux. Une étude plus poussée nous permettrait d'anticiper la rechage de certaines stations de Velibs en fonctions de tous ces facteurs (meteo, lieux touristiques, évènements).")

  st.write("- Le Machine Learning à J+7 pour chaque borne de comptage à chaque horaire de la journée nous permettrait de mettre d'ailleurs en place cette analyse plus complète.")

  st.write("En conclusion, ce dataframe initial nous a permit de nous lancer sur différentes refléxions que nous trouvions pertinentes pour une utilisation ultérieure. Cependant, beaucoup d'autres axes peuvent être encore travaillés grâce à ce thème (préparation des travaux de la ville de Paris sur certaines zones avec création d'itinéraire bis pour les pistes cyclables, préparation de principaux évènements comme les JO 2024 pour alléger les transports en commun et les Ubers....). Tout est envisageable.")

  st.write("Cyclement Votre,")

  st.write("Anne, Karine, Kaity")


  st.balloons()