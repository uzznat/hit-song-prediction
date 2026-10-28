# Guide pour présenter le projet à la prof

Présentation : `presentation/hit_prediction_deck.html` (34 slides, flèches du clavier), PDF à côté.
Pour chaque slide : **le message en une phrase**, puis ce qu'il faut dire. Les chiffres viennent tous du papier
(`paper/main.pdf`) et de `paper_results/`.

---

## Le fil rouge en 30 secondes

On voulait savoir **ce qui prédit vraiment un hit**. On a pris les 13 descripteurs audio de Spotify (78,2 %),
on a ajouté deux signaux simples, **l'historique de l'artiste** et **l'époque** (83,2 %, +5,0 points), puis on a
passé des mois à vérifier que ce gain est réel : autres algorithmes, autres jeux de données, artistes jamais vus,
tests statistiques corrects. En chemin, on a trouvé que plusieurs gains qui avaient l'air énormes étaient des
**artefacts** (les titres de chansons, la FFT, qui détecte en fait le genre). Conclusion : l'historique de
l'artiste est un vrai signal (+2,6 points sur des données indépendantes) ; dans le son lui-même, rien ne résiste
à tous les contrôles.

---

## Partie 1 · Problème et données

**Slide 3 · Le problème.** Les papiers annoncent 86, 88, 89 %, mais on ne peut pas les comparer : chacun a ses
propres données, ils testent sur un seul découpage sans test statistique, et leurs données contiennent des
raccourcis (le modèle peut apprendre *comment les non-hits ont été choisis* au lieu de ce qui fait un hit).

**Slide 4 · Les données.** Quatre jeux de données, et la définition de « hit » change à chaque fois.
- *Spotify Hit Predictor* (Kaggle) : notre jeu principal, 40 560 chansons, moitié hits.
- *ISMIR 2019* : jeu publié et relu par des pairs, sert à vérifier que nos résultats tiennent ailleurs.
- *SpotGenTrack* : 100 000 morceaux, « hit » = popularité Spotify ≥ 55 (seulement 19 % de hits).
- *Billboard fin d'année* : **abandonné**, 2 911 lignes sur 3 397 n'avaient pas de données audio.

## Partie 2 · Méthode

**Slide 5 · Point de départ.** 13 descripteurs audio calculés par Spotify (dansabilité, énergie, tempo...).
C'est de la **classification binaire** : hit ou pas. Une forêt aléatoire sur ces 13 nombres donne **78,2 %**.
C'est la référence de tout le reste.

**Slide 6 · Les deux signaux qu'on ajoute.**
- *L'époque* (la décennie) : un hit des années 60 ne sonne pas comme un hit des années 2010.
- *L'historique de l'artiste* : le nombre de chansons que l'artiste a placées au Hot 100 **dans les décennies
  précédentes uniquement**. « Strictement avant », c'est ce qui garantit qu'on n'utilise aucune info du futur.

**Slide 7 · Les classifieurs, pourquoi, et lequel on garde.**
- **Forêt aléatoire (gardée, modèle principal)** : même configuration que le papier de référence
  (Middlebrook & Sheik), donc comparable ; elle apprend des règles non linéaires (« dans les années 80, tel son
  = hit ») ; on ne la règle **jamais**, pour que tout gain vienne des variables, pas du réglage.
- **XGBoost** : le meilleur en moyenne (84,5 %), mais à égalité avec la forêt sur le découpage canonique
  (p = 0,87). Pas nécessaire.
- **SVM, k-NN, régression logistique** : servent à montrer que le gain ne dépend pas du modèle. Les cinq gagnent
  entre +4,6 et +5,7 points. La régression logistique est la plus faible, parce que les relations ne sont pas
  linéaires.

**Slide 8 · Les autres algorithmes.**
- Extraction de son : *librosa* (185 descripteurs), *FFT* (31 descripteurs spectraux), *PCA* (compresse 185 → 40).
- *TF-IDF + régression logistique* : pour tester si le titre de la chanson aide.
- *TreeSHAP* : explique ce que la forêt utilise (artiste n° 1, décennie n° 2, instrumentalness n° 3).
- Statistiques : *McNemar* (un découpage), *t-test corrigé de Nadeau-Bengio* (25 découpages qui se chevauchent),
  *Bonferroni* (on fait ~60 tests, donc on exige p < 0,0008 pour dire « solide »).

**Slide 9 · Comment on teste chaque gain.** Deux mesures : l'ancien découpage unique (pour reproduire nos
anciens chiffres) et la **validation croisée répétée 5×5** = 25 tests. Le t-test normal est trop optimiste quand
les découpages se chevauchent, d'où la version corrigée.

## Partie 3 · Résultats sur le benchmark

**Slide 10 · Résultat principal.** 78,2 → 83,2 % (+5,0 points), positif dans les 25 découpages, p = 7·10⁻¹⁷.
Le F1 des hits monte aussi (0,787 → 0,831) : précision et rappel montent tous les deux.

**Slide 11 · Pourquoi l'artiste marche.** Un artiste sans hit antérieur : 39 % de ses chansons sont des hits ;
avec 1-2 hits antérieurs : 91 %. Le premier hit change tout.

**Slides 12-14 · Robustesse.** Le gain tient dans **chaque décennie** (+3,7 à +6,6), avec **chaque
classifieur**, et quand on entraîne sur le passé pour prédire une décennie jamais vue (+1,9 à +3,4).

**Slide 15 · Comparaison directe avec Du (2022).** La seule étude académique sur ce jeu. On a refait exactement
son protocole et on retrouve ses chiffres ; en ajoutant notre variable, on fait mieux sur **50 découpages sur 50**.

## Partie 4 · Est-ce que c'est réel ?

**Slide 16 · Le piège.** Le README du jeu Kaggle dit qu'un non-hit doit venir d'un artiste **absent** de la
liste des hits et d'un genre **non-mainstream**. Donc « artiste connu » ≈ « hit » par construction (96,6 %).
Les +5,0 sont gonflés : il faut vérifier sur d'autres données.

**Slides 17-18 · ISMIR 2019.** Protocole A : 76,3 %, au-dessus du meilleur résultat publié (75,04 %).
Mais l'année seule donne déjà 70,2 % (les hits sont plus vieux). Protocole B, où chaque hit est apparié à un
non-hit **de la même année** : l'année devient inutile, l'artiste ajoute encore **+2,6** (p = 3·10⁻⁵).
**C'est le chiffre qu'on défend.**

**Slide 19 · SpotGenTrack.** 81 % de non-hits, donc l'exactitude ne dit rien ; on regarde l'AUC : 0,690 → 0,762.
On trouve 25 % des hits au lieu de 8 %.

**Slides 20-21 · Artistes jamais vus et nouveaux artistes.** Si aucun artiste du test n'est dans
l'entraînement, le gain reste (+5,0). Mais pour un **artiste débutant**, c'est l'époque qui aide (+2,2),
pas l'historique (+0,4, non significatif) — logique, il n'a pas d'historique.

## Partie 5 · Ce qu'on a essayé et enlevé

**Slide 22 · Audit.** Les titres de chansons donnent +6,7 points, plus que tout le reste : **artefact** (les
titres trahissent comment les non-hits ont été collectés). Le son riche (librosa) : +3,8 sur un découpage,
+0,9 non significatif sur 25 : **c'était de la chance**.

**Slide 23 · FFT = raccourci de genre.** Sur 35 000 chansons, la FFT ajoute +1,4 (très significatif). Mais les
non-hits sont choisis dans des genres de niche, et le son permet de reconnaître le genre. Si on donne le genre
au modèle, la FFT fait *baisser* l'exactitude ; sur un échantillon équilibré par genre, elle n'apporte rien
(+0,1, p = 0,84). **Enlevée.** Le genre lui-même n'est pas utilisé comme variable : Spotify l'attribue après
coup, ce serait tricher.

**Slide 24 · Vieillissement du modèle.** On entraîne sur une décennie et on teste plus tard. Le son seul passe
de 0,809 à 0,689 trois décennies plus tard ; l'historique de l'artiste vieillit bien ; la FFT vieillit mal.

**Slide 25 · Trois vérifications.** (1) 99,99 % des hits sont bien retrouvés dans l'historique : pas de
problème d'appariement. (2) Les paroles aident un peu sur SpotGenTrack (AUC +0,023) : piste prometteuse.
(3) Les métriques F1 confirment les gains.

**Slide 26 · Tableau récapitulatif des variables.** C'est la slide à garder sous les yeux pour les questions
« vous avez gardé quoi, enlevé quoi, pourquoi ».

**Slide 27 · Statistiques.** ~60 tests, donc certains passent par hasard. Bonferroni : tous les résultats sur
lesquels repose la conclusion restent significatifs.

**Slide 28 · Honnêteté.** Huit affirmations de nos anciennes slides étaient fausses ; on les a corrigées.

## Partie 6 · Séries temporelles (le plan de lecture de la prof)

**Slide 29 · Composantes et prévision.** Sur 68 ans de classements hebdomadaires :
- *Nombre de nouvelles entrées par semaine* : surtout une **tendance** (11 en 1959, 5,5 en 2001, 11 avec le
  streaming), saisonnalité faible (janvier calme), un pic dû à un **changement de règle** en décembre 1998.
- *Chansons de Noël* : saisonnalité pure ; la **FFT trouve seule le cycle de 52 semaines** et ses harmoniques.
- Prévision à 12 mois : **ARIMA, ARIMA + Fourier et Holt-Winters** sont les meilleurs (gardés) ;
  SARIMA n'aide pas (pas de saison à modéliser) ; le machine learning fait moins bien (800 points, c'est trop
  peu, et les arbres ne savent pas prolonger une tendance).

**Slide 30 · Séries temporelles pour prédire un hit.** Deux expériences, deux réponses opposées.
- *Après la sortie* : avec les **3 premières semaines** d'une chanson au Hot 100 (positions, meilleure place,
  pente, accélération) + l'année et l'historique de l'artiste, une forêt aléatoire prédit si elle atteindra le
  **top 10** avec une AUC de **0,89** (0,92 avec 4 semaines) ; l'année et l'artiste seuls ne donnent que 0,59.
  Ce n'est pas trivial : sur les chansons **pas encore** dans le top 10 après 3 semaines, AUC 0,87 ; entraîné
  avant 2000 et testé après, 0,85. **Gardé.** Attention : ici toutes les chansons sont déjà entrées au
  classement, donc la question est « gros hit ou petit hit », pas « hit ou flop ».
- *Avant la sortie* : on mesure si le son d'une chanson suit la **tendance du marché** (distance aux hits de la
  décennie précédente, et direction de la tendance). Sur le modèle principal : +0,1 point, non significatif.
  Sur le son seul : +1,8, mais la décennie seule donne +2,0 : la tendance ne transporte que l'**époque**, que le
  modèle a déjà. **Enlevé.**
- Message : les séries temporelles sont le meilleur prédicteur qu'on ait trouvé **une fois la chanson sortie** ;
  avant la sortie, elles n'ajoutent rien à l'historique de l'artiste et à l'époque.

## Conclusion

**Slide 31 · Ce qui est nouveau.** L'idée de l'historique d'artiste existait (Interiano 2018, Middlebrook
2019). Ce qu'on apporte : la version sans fuite d'information, les tests statistiques, la comparaison directe,
la validation externe et l'audit des raccourcis.

**Slide 32 · Limites.** Artistes appariés par nom ; pas de deep learning ; l'historique mesure en partie
« l'avantage cumulatif » (un artiste connu est plus diffusé), pas la qualité de la chanson.

**Slide 33 · À retenir.** +5,0 sur le benchmark (borne haute, gonflée), **+2,6 sur des données
indépendantes** (le chiffre honnête), 76,3 % au-dessus du meilleur publié. Savoir quels gains sont réels
compte plus que le chiffre affiché.

**Slide 34 · Ce qu'on a réussi à prouver.** La slide de fin, à laisser affichée pendant les questions.
- *Prouvé* : (1) l'historique de l'artiste est un vrai signal (+5,0 sur le benchmark, +2,6 sur des données
  indépendantes) ; (2) il bat les travaux publiés (76,3 % contre 75,04 %, et Du sur 50 découpages sur 50) ;
  (3) ce n'est pas un effet d'un modèle ou des artistes connus ; (4) pour un nouvel artiste, c'est l'époque qui
  aide ; (5) après la sortie, les 3 premières semaines prédisent un top 10 (AUC 0,89).
- *Réfuté* : titres (artefact), son riche (chance), FFT (genre), année sur ISMIR (raccourci de date), tendance
  du marché (ne porte que l'époque). Dans le son lui-même, rien ne résiste à tous les contrôles.
- Phrase de fin : ce qui prédit un hit, c'est surtout **qui chante et quand**, et une fois sortie, **comment se
  passent ses premières semaines**.

---

## Questions probables de la prof

- **« Pourquoi pas du deep learning ? »** Notre question porte sur les *entrées* (quelles variables portent un
  signal), à modèle égal. Les cinq classifieurs donnent le même gain. Un modèle appris sur le son brut reste une
  piste future.
- **« Pourquoi la forêt aléatoire et pas XGBoost, qui est meilleur ? »** Pour être comparable au papier de
  référence, et parce que la différence n'est pas significative sur le découpage canonique.
- **« L'historique de l'artiste, ce n'est pas de la triche ? »** Non : on ne compte que les décennies
  *précédentes*, information disponible avant la sortie. Mais sur Kaggle le gain est gonflé par la construction
  du jeu ; c'est pour ça qu'on donne +2,6 (ISMIR, années appariées) comme chiffre honnête.
- **« Et pour un nouvel artiste ? »** L'historique n'aide pas (normal) ; l'époque aide un peu. C'est la
  principale limite, et la suite naturelle du projet.
- **« Les séries temporelles servent à quoi pour votre problème ? »** À prédire la suite une fois la chanson
  sortie (AUC 0,89 pour le top 10 avec 3 semaines). Avant la sortie, la tendance du marché n'ajoute rien
  au-delà de l'époque.
- **« Pourquoi avoir enlevé la FFT si elle est significative ? »** Parce qu'elle reconnaît le genre, et le genre
  a servi à choisir les non-hits. Sur des données équilibrées par genre, elle n'apporte plus rien.
