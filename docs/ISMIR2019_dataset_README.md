# Hit Song Prediction Dataset

This dataset is based on the Million Song Dataset (MSD), which contains one million songs that are representative for western commercial music released between 1922 and 2011. The dataset contains release year information for 515,576 of the MSD songs. Please refer to http://millionsongdataset.com/ for further information on the million song dataset.

For our hit song prediction experiments, we extract high- and low-level audio features using the Essentia toolkit (cf. https://essentia.upf.edu/). For the high-level features, we make use of the pre-trained classifiers as provided by Essentia. For a detailed description of the features, please visit the Essentia documentation.


The dataset hence contains:
* audio features: the compressed file `msd_audio_features.tar.gz` contains the low- and high-level features for each track, stored as json files. Please note that we organize all MSD audio feature files based on the track's identifier with one folder holding all tracks with the same first letter of the track identifier to keep the files manageable. For each track, we provide two files: one containing the high-level and one containing the low-level features extracted by Essentia.
* billboard data: the folder billboard_data contains two files: `msd_bb_matches.csv` contains information about the MSD tracks that were also featured in the Billboard Hot 100 charts. Here, we provide the MSD id, Echo Nest id, artist name, track title, release year, peak position in Billboard charts and the number of weeks in the charts. The second file, `msd_bb_non_matches.csv` contains meta-information about the tracks of the MSD that were not featured in the Billboard Hot 100 and hence were used as negative samples. Here, we provide the MSD id, Echo Nest id, artist name, track title and the release year.


If you make use of the dataset, please kindly cite the following paper:

Eva Zangerle, Michael Vötter, Ramona Huber, and Yi-Hsuan Yang. Hit Song Prediction: Leveraging Low- and High-Level Audio Features. In Proceedings of the 20th International Society for Music Information Retrieval Conference 2019 (ISMIR 2019), 2019.

```
@inproceedings{zangerle_ismir19,
title = {{Hit Song Prediction: Leveraging Low- and High-Level Audio Features}},
author = {Eva Zangerle and Michael V\"{o}tter and Ramona Huber and Yi-Hsuan Yang},
year = {2019},
booktitle = {{Proceedings of the 20th International Society for Music Information Retrieval Conference 2019 (ISMIR 2019)}},
}
```

Contact:
Eva Zangerle
Department of Computer Science
University of Innsbruck, Austria
