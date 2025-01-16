# Classical Music Machine Learning

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Acknowledgments

Thanks to [Classical Archives](https://www.classicalarchives.com/midi.html) and [kunstderfuge](https://kunstderfuge.com/) for the midi files

## Requirements

-   Python 3.9–3.12
-   Installing packages from requirements.txt

## Training

To train, run the **lstm.py** file.

The network will use all the midi files in the ./music directory. It's best if the midi files only contain one instrument.

## Generating

Once you finish training, you can generate text by running the **predict.py** file.

You can run the prediction file right away using the **weights.hdf5** file
