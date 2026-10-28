# Guide pour présenter le projet à la prof

Présentation : `presentation/hit_prediction_deck.html` (flèches du clavier), PDF à côté.

- **Slides 1 à 19 : l'histoire du projet**, à raconter dans l'ordre. Chaque slide se termine par une flèche
  bleue « → ... » : c'est la phrase qui fait passer à la suivante (« donc on a fait ça »).
- **Slides 20 à 44 : l'annexe**, pour les questions (tableaux d'algorithmes, de variables, tous les chiffres).

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

## Slide par slide (ce qu'il faut dire)

**1 · Titre.** « On va vous raconter notre projet : ce qu'on a essayé, ce qui a marché, et ce qui était un piège. »

**2 · La question.** Un hit = une chanson entrée au Billboard Hot 100. L'ordinateur apprend sur des chansons dont
on connaît la réponse, puis on le teste sur des chansons qu'il n'a jamais vues. Au hasard, on aurait 50 %.

**Chapitre 1 · Le son**

**3 · On a commencé par le son.** Spotify décrit chaque chanson avec 13 nombres (dansabilité, énergie, tempo...).
Résultat : **78 %**. Mieux qu'une pièce, mais 1 chanson sur 5 est encore fausse.

**4 · Comment l'ordinateur apprend et comment on teste.** Forêt aléatoire = 80 petits arbres de décision qui posent
des questions simples et votent. On ne la règle jamais, pour que toute amélioration vienne des indices qu'on lui
donne. On teste 25 fois avec des chansons cachées différentes, et un test statistique dit si l'amélioration est
réelle.

**Chapitre 2 · Qui et quand**

**5 · Notre idée.** Un hit, ce n'est pas que du son. *Qui* : l'artiste a-t-il eu des hits **avant** (seulement
les décennies précédentes, jamais le futur) ? *Quand* : la décennie, car un hit de 1965 ne sonne pas comme un
hit de 2015.

**6 · Résultat.** 78 → 83 % (+5 points), mieux dans les 25 tests sur 25.

**7 · Pourquoi « qui » marche.** Artiste sans hit avant : 39 % de ses chansons sont des hits. Avec 1 ou 2 hits
avant : 91 %.

**Chapitre 3 · Est-ce de la chance ?**

**8 · On a essayé de casser le résultat.** 5 algorithmes différents : tous +4,6 à +5,7. Chaque décennie :
+3,7 à +6,6. Prédire une décennie jamais vue : +1,9 à +3,4. La seule autre étude (Du, 2022), refaite à
l'identique : on fait mieux 50 fois sur 50.

**Chapitre 4 · Un piège dans les données**

**9 · Le piège.** Les règles du jeu de données : un non-hit doit venir d'un artiste **sans hit** cette décennie,
dans un genre de niche. Donc « l'artiste a eu des hits » reflète en partie la règle, pas la musique (97 % des
chansons d'artistes déjà connus sont des hits ici). Notre +5 est trop optimiste.

**10 · Un autre jeu de données (ISMIR 2019).** Les non-hits y sont des chansons au hasard. On obtient **76,3 %**,
mieux que le meilleur résultat publié (75,0 %). Mais nouveau piège : l'année seule donne déjà 70 %, parce que
les hits de ce jeu sont des chansons plus anciennes.

**11 · Le test équitable.** Chaque hit contre un non-hit **de la même année**. L'année devient inutile ; l'artiste
ajoute encore **+2,6 points** (10 tests sur 10). C'est **notre chiffre honnête**.

**12 · Nouveaux artistes.** Si aucun artiste du test n'a été vu à l'entraînement, le gain reste le même (+5,0) :
l'ordinateur ne mémorise pas des noms. Mais pour un **tout nouvel artiste** (première chanson), « qui » ne peut pas
aider (+0,4, rien) ; « quand » aide un peu (+2,2). C'est le cas le plus difficile.

**Chapitre 5 · Ajouter plus, et attraper les faux gains**

**13 · Plus d'indices.** Titres des chansons : +6,7, trop beau pour être vrai, les titres trahissent comment le jeu
a été collecté → enlevé. Son détaillé (185 mesures) : +3,8 sur un test, mais +0,9 sur 25 tests → c'était de la
chance → enlevé. Petits détails (longueur du titre, refrain) : rien → enlevés.

**14 · La FFT.** Elle découpe le son en fréquences (basses, médiums, aigus) : +1,4 point, très significatif. Mais
les non-hits viennent de genres de niche, et le son révèle le genre : l'ordinateur reconnaissait le jazz ou la
musique latine, pas les hits. Test équitable (autant de hits que de non-hits dans chaque genre) : +0,1, rien →
enlevée. Leçon : un gros gain peut être un piège.

**Chapitre 6 · Séries temporelles (le plan de lecture de la prof)**

**15 · Lire le classement comme un signal.** Une série temporelle se découpe en tendance (changement lent),
saisonnalité (se répète chaque année) et résidu. Le nombre de nouvelles chansons par semaine est surtout une
tendance ; les chansons de Noël sont de la saisonnalité pure, et la FFT trouve seule le cycle de 52 semaines.
Pour prévoir l'année suivante, ARIMA et Holt-Winters sont les meilleurs ; le machine learning ne fait pas mieux
(800 points, c'est trop peu).

**16 · Les séries temporelles prédisent-elles un hit ?** Ça dépend de **quand** on pose la question.
- **Avant la sortie** (notre question) : est-ce que la chanson ressemble aux hits récents ? +0,1, rien. La tendance
  ne connaît que la décennie, qu'on avait déjà. **Non.**
- **Après la sortie** (une autre question) : la chanson vient d'entrer au classement. Ses 3 premières semaines
  disent si elle atteindra le top 10 : quand on compare deux chansons, l'ordinateur les classe bien 89 fois sur
  100 (AUC 0,89). **Oui.**
- Les deux sont vrais en même temps : pas d'aide pour deviner un hit avant sa sortie, mais le meilleur indice une
  fois qu'il est sorti.

**Fin**

**17 · Ce qu'on a prouvé.** Vrai : qui chante compte (+5, et +2,6 sur un test équitable) ; on bat les résultats
publiés ; ça marche avec tous les algorithmes et pour des artistes jamais vus ; pour un tout nouvel artiste, seule
l'époque aide un peu ; après la sortie, les 3 premières semaines prédisent le top 10. Faux (pièges) : titres, son
détaillé, FFT, année, tendances. Dans le son lui-même, rien n'a résisté à tous les tests. Phrase de fin : un hit,
c'est surtout **qui chante et quand**, et une fois sorti, **comment se passent ses premières semaines**.

**18 · Ce qu'on ne sait pas encore faire.** Les tout nouveaux artistes ; les paroles et le deep learning ; un jeu de
données sans piège.

**19 · Merci, questions.** Les slides suivantes sont les détails.

---

## L'annexe (slides 20 à 44), pour répondre aux questions

| Si la prof demande... | Slide |
|---|---|
| Les jeux de données | 21 |
| Quels algorithmes, pourquoi, lequel on garde | 22 |
| Les autres algorithmes (librosa, FFT, PCA, TF-IDF, SHAP, tests statistiques) | 23 |
| Comment on teste (validation croisée, test corrigé) | 24 |
| Chaque décennie, chaque classifieur, prédire le futur, Du | 25 à 28 |
| ISMIR (protocoles A et B), SpotGenTrack | 29 à 31 |
| Artistes jamais vus, nouveaux artistes | 32 et 33 |
| Les faux gains, la FFT et le genre | 34 et 35 |
| Vieillissement du modèle, appariement des artistes, paroles, F1 | 36 et 37 |
| **Toutes les variables : gardée ou enlevée, et pourquoi** | **38** |
| Statistiques (Bonferroni), ce qu'on a corrigé | 39 et 40 |
| Ce qui est nouveau par rapport aux autres, limites | 41 et 42 |
| Séries temporelles en détail | 43 et 44 |

---

## Questions probables de la prof

- **« Pourquoi pas du deep learning ? »** On cherchait quelles *entrées* portent un signal, à modèle égal. Les cinq
  classifieurs donnent le même gain. Le deep learning sur le son brut reste une piste future.
- **« Pourquoi la forêt aléatoire et pas XGBoost, qui est meilleur ? »** Pour être comparable au papier de
  référence, et la différence n'est pas significative sur le découpage canonique (p = 0,87).
- **« L'historique de l'artiste, ce n'est pas de la triche ? »** Non : on ne compte que les décennies *précédentes*,
  connues le jour de la sortie. Mais sur le jeu Kaggle le gain est gonflé par sa construction ; c'est pour ça qu'on
  donne +2,6 (test équitable sur ISMIR) comme chiffre honnête.
- **« Et pour un nouvel artiste ? »** L'historique ne peut pas aider ; l'époque aide un peu. C'est la principale
  limite et la suite naturelle.
- **« Pourquoi avoir enlevé la FFT si elle est significative ? »** Parce qu'elle reconnaît le genre, et le genre a
  servi à choisir les non-hits. Avec des genres équilibrés, elle n'apporte plus rien.
- **« Les séries temporelles servent à quoi pour votre problème ? »** Pas à deviner un hit avant sa sortie (+0,1).
  Mais une fois la chanson sortie, ses premières semaines prédisent le top 10 (AUC 0,89).
