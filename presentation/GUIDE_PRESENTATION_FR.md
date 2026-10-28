# Guide pour présenter le projet à la prof

Présentation : `presentation/hit_prediction_deck.html` (42 slides, flèches du clavier), PDF à côté.

C'est **une seule histoire**, du début à la fin : les slides simples racontent ce qu'on a fait, et les slides de
détails (graphiques, tableaux) arrivent au moment où elles se passent dans l'histoire. Chaque slide se termine par
une flèche bleue « → ... » : c'est la phrase qui fait passer à la suivante (« donc ensuite on a fait ça »).
Il suffit de lire la flèche à voix haute pour enchaîner.

Les chiffres viennent tous du papier (`paper/main.pdf`) et de `paper_results/`.

---

## L'histoire en une minute

On voulait savoir si un ordinateur peut deviner qu'une chanson sera un hit. On a commencé par **le son** :
78 % de bonnes réponses. Ensuite on a eu l'idée d'ajouter **qui chante** (l'artiste a-t-il déjà eu des hits ?)
et **quand** (la décennie) : 83 %. On a essayé de casser ce résultat, il a tenu. Puis on a découvert un
**piège** dans les données : les non-hits avaient été choisis parmi des artistes sans hit. On a donc testé sur
d'autres données, avec un test équitable : il reste **+2,6 points**, notre chiffre honnête. Ensuite on a essayé
d'ajouter plus de son ; à chaque fois le gain était un **piège** (titres, chance, genre). Enfin la prof nous a fait
étudier les **séries temporelles** : elles n'aident pas avant la sortie, mais après la sortie les 3 premières
semaines prédisent très bien le top 10.

---

## Slide par slide

### La question (slides 1 à 3)

**1 · Titre.** « On va vous raconter notre projet : ce qu'on a essayé, ce qui a marché, et ce qui était un piège. »

**2 · La question.** Un hit = une chanson entrée au Billboard Hot 100. L'ordinateur apprend sur des chansons dont
on connaît la réponse, puis on le teste sur des chansons qu'il n'a jamais vues. Au hasard : 50 %.

**3 · Les quatre jeux de données** (tableau). Kaggle (40 560 chansons, notre jeu principal), ISMIR 2019 (publié,
sert de vérification), SpotGenTrack (100 000 morceaux, peu de hits), et le Billboard de fin d'année, abandonné
parce que 2 911 lignes sur 3 397 n'avaient pas de données audio.

### Chapitre 1 · Le son (slides 4 à 8)

**4 · On a commencé par le son.** 13 nombres Spotify par chanson → **78 %**. Une chanson sur 5 encore fausse.

**5 · Comment l'ordinateur apprend.** Forêt aléatoire = 80 petits arbres qui posent des questions et votent. On
teste 25 fois avec des chansons cachées différentes.

**6 · Les classifieurs** (tableau). La forêt aléatoire est le modèle principal : même configuration que le papier
de référence, jamais réglée. XGBoost est le meilleur (84,5 %) mais pas significativement mieux. SVM, k-NN et
régression logistique servent à montrer que le gain ne dépend pas du modèle.

**7 · Les autres algorithmes.** Pour mesurer le son (librosa, FFT, PCA, Essentia), pour le texte et l'explication
(TF-IDF, SHAP), et pour tester (McNemar, t-test corrigé, Bonferroni).

**8 · Comment on teste chaque gain.** Un ancien découpage unique, et surtout 25 découpages (validation croisée
5×5) avec un t-test corrigé, parce que le t-test normal est trop optimiste quand les découpages se chevauchent.

### Chapitre 2 · Qui et quand (slides 9 à 11)

**9 · Notre idée.** *Qui* : l'artiste a-t-il eu des hits **avant** (seulement les décennies précédentes) ?
*Quand* : la décennie.

**10 · Résultat** (graphique). 78 → 83 % (+5 points), mieux dans 25 tests sur 25.

**11 · Pourquoi « qui » marche** (graphique). Sans hit avant : 39 % de hits. Avec 1 ou 2 hits avant : 91 %.

### Chapitre 3 · Est-ce de la chance ? (slides 12 à 16)

**12 · On a essayé de casser le résultat** : vue d'ensemble des 4 vérifications.

**13 · Chaque décennie** (graphique) : +3,7 à +6,6.

**14 · Chaque classifieur** (graphique) : +4,6 à +5,7. On garde la forêt aléatoire. SHAP : l'artiste est la
variable n° 1.

**15 · Prédire le futur** (graphique) : entraîné sur les anciennes décennies, testé sur une décennie jamais vue :
+1,9 à +3,4.

**16 · Contre la seule autre étude (Du, 2022)** (graphique) : on refait sa méthode, on retrouve ses chiffres, et
avec notre variable on fait mieux 50 fois sur 50.

### Chapitre 4 · Un piège dans les données (slides 17 à 25)

**17 · Le piège.** Un non-hit doit venir d'un artiste **sans hit** cette décennie, dans un genre de niche. 97 % des
chansons d'artistes déjà connus sont des hits ici : notre +5 est trop optimiste.

**18 · Un autre jeu de données (ISMIR).** 76,3 %, mieux que le meilleur publié (75,0 %). Mais l'année seule donne
déjà 70 % : les hits y sont plus anciens.

**19 · Les chiffres d'ISMIR** (graphique) : année seule 70,2 %, son 70,8 %, son + année 73,9 %, son + année +
artiste 76,3 %.

**20 · Le test équitable.** Chaque hit contre un non-hit de la même année : l'année devient inutile, l'artiste
ajoute encore **+2,6** — notre chiffre honnête.

**21 · Les chiffres du test équitable** (graphique) : +2,6, p = 3·10⁻⁵, 10 tests sur 10.

**22 · SpotGenTrack** (graphiques). 81 % de non-hits, donc on regarde l'AUC : 0,690 → 0,777. On trouve 25 % des
hits au lieu de 8 %.

**23 · Nouveaux artistes** : vue d'ensemble.

**24 · Artistes jamais vus** : le gain reste (+5,0 ; +2,9 sur ISMIR équitable ; 75,9 % toujours au-dessus du
publié).

**25 · Tout nouveaux artistes** (tableau) : l'historique ne peut pas aider (+0,4), l'époque aide (+2,2).

### Chapitre 5 · Ajouter plus, et attraper les faux gains (slides 26 à 32)

**26 · Plus d'indices** : titres (artefact), son détaillé (chance), petits détails (rien). Tous enlevés.

**27 · Les chiffres** (graphique) : un seul test contre 25 tests. Le son détaillé passe de +3,8 à +0,9.

**28 · La FFT** : elle avait l'air bonne (+1,4), mais elle reconnaissait le genre.

**29 · Le test du genre** (graphique) : avec le genre connu, la FFT fait baisser ; avec des genres équilibrés,
+0,1. Le genre lui-même n'est pas utilisé comme variable : Spotify l'attribue après coup.

**30 · Le modèle vieillit** (tableau) : le son seul passe de 0,809 à 0,689 trois décennies plus tard ;
l'historique de l'artiste vieillit bien, la FFT vieillit mal.

**31 · Trois vérifications** : 99,99 % des hits retrouvés (pas de problème d'appariement) ; les paroles aident un
peu (prometteur) ; les F1 confirment les gains.

**32 · Tableau récapitulatif de toutes les variables** : gardée, enlevée, contrôle ou prometteuse, et pourquoi.
**La slide à garder en tête pour les questions.**

### Chapitre 6 · Séries temporelles, le plan de lecture de la prof (slides 33 à 36)

**33 · Lire le classement comme un signal** : tendance, saisonnalité, résidu ; ARIMA est le meilleur pour prévoir.

**34 · Les détails** (tableau) : nouvelles chansons par semaine = surtout une tendance ; chansons de Noël =
saisonnalité pure, la FFT trouve le cycle de 52 semaines. ARIMA, ARIMA + Fourier, Holt-Winters gardés ;
SARIMA et machine learning enlevés.

**35 · Les séries temporelles prédisent-elles un hit ?** **Avant la sortie : non** (+0,1, la tendance ne connaît
que la décennie). **Après la sortie : oui** (les 3 premières semaines prédisent le top 10, AUC 0,89). Ce sont
deux questions différentes, et les deux réponses sont vraies.

**36 · Les détails des deux expériences** (tableau) : AUC 0,59 sans les semaines, 0,75 / 0,85 / 0,89 / 0,92 avec
1 / 2 / 3 / 4 semaines ; 0,87 sur les chansons pas encore dans le top 10 ; 0,85 en testant après 2000.

### La fin (slides 37 à 42)

**37 · Statistiques** : ~60 tests, Bonferroni ; tous les résultats importants restent solides.

**38 · Ce qu'on a corrigé** : 8 affirmations de nos anciennes slides étaient fausses.

**39 · Ce qui est nouveau** : l'idée de l'historique d'artiste existait (Interiano 2018, Middlebrook 2019) ; ce
qu'on apporte, ce sont les tests, la comparaison directe, la validation externe et l'audit des pièges.

**40 · Ce qu'on a prouvé.** Phrase de fin : un hit, c'est surtout **qui chante et quand**, et une fois sorti,
**comment se passent ses premières semaines**.

**41 · Limites et suite** : nouveaux artistes, deep learning, paroles, un jeu de données sans piège.

**42 · Merci, questions.**

---

## Questions probables de la prof

- **« Pourquoi pas du deep learning ? »** On cherchait quelles *entrées* portent un signal, à modèle égal. Les cinq
  classifieurs donnent le même gain (slide 14). Le deep learning sur le son brut reste une piste future.
- **« Pourquoi la forêt aléatoire et pas XGBoost, qui est meilleur ? »** Pour être comparable au papier de
  référence, et la différence n'est pas significative (slide 6).
- **« L'historique de l'artiste, ce n'est pas de la triche ? »** Non : on ne compte que les décennies *précédentes*,
  connues le jour de la sortie. Mais sur Kaggle le gain est gonflé par la construction du jeu (slide 17) ; c'est
  pour ça qu'on donne +2,6 (slide 21) comme chiffre honnête.
- **« Et pour un nouvel artiste ? »** L'historique ne peut pas aider ; l'époque aide un peu (slide 25). C'est la
  principale limite.
- **« Pourquoi avoir enlevé la FFT si elle est significative ? »** Elle reconnaît le genre, et le genre a servi à
  choisir les non-hits. Avec des genres équilibrés, elle n'apporte rien (slide 29).
- **« Les séries temporelles servent à quoi pour votre problème ? »** Pas à deviner un hit avant sa sortie ; mais
  une fois la chanson sortie, ses premières semaines prédisent le top 10 (slides 35-36).
