# Modélisation de séries temporelles — trajets Vélib’ par heure

Projet réalisé par **Nour Bouabid et Marya Khelifa**.

## Objectif

Analyser l’évolution horaire du **nombre de trajets Vélib’** et étudier l’ajustement de modèles autorégressifs à moyenne mobile (**ARMA**) après traitement de la tendance et de la saisonnalité. Le notebook étend ensuite cette démarche à la durée moyenne des trajets et à la proportion de vélos électriques.

## Données

Le notebook lit **`velib_horaire.csv`**, convertit `Departure_Time` en dates et place cette colonne en index chronologique. Trois séries sont présentes :

- `nombre_trajets` : nombre de trajets par heure ;
- `duree_moyenne_sec` : durée moyenne d’un trajet, en secondes ;
- `proportion_velos_electriques` : proportion de trajets effectués avec un vélo électrique.

Les sorties conservées portent sur **1 440 observations horaires** du nombre de trajets, avec une moyenne d’environ **6 016 trajets par heure**. Les données proviennent du site Kaggle.


## Méthodologie

1. **Analyse descriptive** de la série et recherche d’une tendance polynomiale par régression OLS.
2. **Étude des saisonnalités** à l’aide de l’autocorrélation (ACF), ainsi que des profils horaires et hebdomadaires. Le notebook relève des cycles journaliers et hebdomadaires.
3. **Différenciation** de la série `nombre_trajets` aux retards **1, 24 et 168 heures** pour constituer `Y_model`.
4. **Étude ACF/PACF** de la série transformée, puis ajustement de modèles **ARMA(p, q)** à l’aide de `statsmodels` (implémentation `ARIMA(p, 0, q)`, avec `p` et `q` de 0 à 4).
5. **Sélection par AIC** parmi les ajustements retenus comme convergents, diagnostic des résidus et prévisions sur une partie finale de `Y_model`.
6. **Application aux autres séries** au moyen de la fonction `etude_complete()` importée depuis un module local `etude.py`.


## Limites

Le modèle est choisi sur une série transformée, et les erreurs rapportées concernent cette série. La qualité des prévisions sur l’échelle originale du nombre de trajets reste **à évaluer** après inversion des différenciations et contrôle des résidus.
