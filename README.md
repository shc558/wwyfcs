## What Would Your Favorite Character Say?
WWYFCS is a deep learning based chatbot that impersonates Game of Thrones characters. Trained on the script of the beloved TV series, the chatbot is able to speak from the perspective of any character of your liking (technically, but characters with more data to train may perform better).



### How was it built?
WWYFCS was built on Microsoft's [DialoGPT](https://github.com/microsoft/DialoGPT), a GPT-2 based model pre-trained on 147M of Reddit discussion threads. During the construction of the training dataset, I tagged each line with respective speaker's name, so that the trained model can be prompted to generate character-specific responses.



### Installation

#### Requirements
This repository assumes the use of Python 3.8. GPU is recommended for model training.

```
git clone https://github.com/shc558/wwyfcs.git
cd wwyfcs
```


#### Dependencies
In a virtual environment or a [PyTorch VM instance](https://cloud.google.com/ai-platform/deep-learning-vm/docs/pytorch_start_instance):

```
pip install -r requirements.txt
```


### Generating training data

The Game of Thrones script can be downloaded [here](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons/download).

```
python wwyfcs/utils/create_examples.py \
--file_path [path/to/raw/dataset] (required)
--data_colname Sentence (required)
--id_colname Name (required)
--output_path (optional)
--character (optional, specific character to extract)
--len_context (default=9, # previous responses to use as context)
--eval_size (default=0.1, fraction of data to use for evaluation)
```


### Training

```
python wwyfcs/trainer/train_language_model.py \
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


### Running the app using Docker

After [Docker](https://www.docker.com/products/docker-desktop) is installed, pull and run the image using:

```
docker run -p 8501:8501 -ti shc558/wwyfcs_app:v1
```
When the Streamlit app is up and running, input text from your **terminal** to start chatting.


#### Chatting with [Jon Snow (beta)](https://bot.dialogflow.com/jon-snow )
Note: bot will generate empty responses while model is loading.
