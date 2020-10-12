# What Would Your Favorite Character Say?
WWYFCS is a deep learning based chatbot that impersonates Game of Thrones characters. Trained on the script of the beloved TV series, the chatbot is able to speak from the perspective of any character of your liking (technically).


# How was it built?
WWYFCS was built on DialoGPT [citation], a GPT-2 based model pre-trained on millions of Reddit discussion threads. During the construction of the training dataset, I tagged each line with the respective speaker's name, so that the trained model can be prompted to generate character-specific responses.

# Installation

## Requirements
This repository assumes the use of Python 3.8. GPU is recommended for model training.

```
git clone https://github.com/shc558/wwyfcs.git
cd wwyfcs
```

## Dependencies
In a virtual environment or a PyTorch VM instance:

```
pip install -r requirements
```

# Generating training data

The Game of Thrones script can be downloaded [here](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons/download).

```
python utils/create_examples.py \
--file_path [path/to/raw/dataset] (required)
--data_colname Sentence (required)
--id_colname Name (required)
--output_path (optional)
--character (optional)
--len_context (default = 9)

```

# Model training example

```
python trainer/train_language_model \
--output_dir [path/to/save/model/outputs] (required)
--train_data_file [path/to/train/dataset] (required)
--eval_data_file [path/to/eval/dataset] (required)
--do_train
--do_eval
--overwrite_output_dir
--per_device_train_batch_size 4 (default=8)
--per_device_eval_batch_size 4 (default=8)
--num_train_epochs 3
```

Use -h to see all arguments.

To train on Colab, see [Fine_tunining_DialoGPT](https://github.com/shc558/wwyfcs/blob/dev/notebooks/Fine_tuning_DialoGPT.ipynb).
