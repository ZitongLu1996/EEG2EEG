# EEG2EEG

Generate your neural signals from mine: individual-to-individual EEG converters

## Requirements

- Python 3.6, or later
- pytorch
- scipy
- scikit-learn
- tqdm
- matplotlib
- pandas
- PIL
- neurora

## Models

- EEG2EEG Model
- NoConnection Model
- NoCosineLoss Model
- Linear Model

## Dataset

THINGSEEG2 - you can get data [here](https://osf.io/3jk45/).

## File Structure

`getdata/.` folder: Get data to use from raw data.

`models/.` folder: Models files to run.

`analysis/.` folder: Analysis scripts.

## Citation

If you are interested in our work, please consider citing the following:
```
@article{Lu&Golomb_CogSci_2023,
  title = {Generate your neural signals from mine: individual-to-individual EEG converters},
  author = {Lu, Zitong and Golomb, Julie D.},
  booktitle = {Proceedings of the 45th Annual Meeting of the Cognitive Science Society (CogSci 2023)},
  doi = {10.48550/arXiv.2304.10736},
  url = {https://doi.org/10.48550/arXiv.2304.10736},
}

```

This project is under the MIT license. For further questions, please contact <strong><i>Zitong Lu</i></strong> at [lu.2637@osu.edu](mailto:lu.2637@osu.edu).
