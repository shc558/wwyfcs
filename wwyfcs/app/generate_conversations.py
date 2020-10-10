# -*- coding: utf-8 -*-
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
from PIL import Image


tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')

st.title('WWYFCS')
st.header('What would your favorite character say?')
total_steps = st.sidebar.slider(label='Turns of conversation',min_value=1,max_value=10, value=5)
top_k= st.sidebar.slider(label='Level of randomness',min_value=0,max_value=30, value=0)

character = st.selectbox('Select a character',options = [
'',
'DialoGPT',
'Jon Snow',
'Arya Stark',
'Daenerys Targaryen',
'Tyrion Lannister',
'Cersei Lannister',
'Missandei',
'Hodor'
])


@st.cache
def load_model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name)

if character == '':
    st.write('Please select a character to begin.')

elif character=='DialoGPT':
    model_name='microsoft/DialoGPT-small'
    prompt_ids = ''
    model =  load_model(model_name)
    # Let's chat
    for step in range(total_steps):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input = input(">> User:")
        st.write('User:',new_user_input)
        new_user_input_ids = tokenizer.encode(new_user_input + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1500,
        do_sample=top_k > 0,
        top_p=0.95,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
        )
        # pretty print last ouput tokens from bot
        st.write("{}: {}".format(character, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


else:
    image = Image.open(os.path.join(os.getcwd(),'app','images','{}.jpg'.format(character)))

    st.image(image, width=100, use_column_width=False)

    model_name=os.path.join(os.getcwd(),'app','model_n4')
    prompt_ids = tokenizer.encode(character.lower()+':', return_tensors='pt')
    model =  load_model(model_name)

    # Let's chat
    for step in range(total_steps):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input = input(">> User:")
        st.write('User:',new_user_input)

        new_user_input_ids = tokenizer.encode(new_user_input + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids, prompt_ids], dim=-1) if step > 0 else torch.cat([new_user_input_ids, prompt_ids], dim=-1)

        # generated a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1500,
        do_sample=top_k > 0,
        top_p=0.95,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
        )
        # pretty print last ouput tokens from bot
        st.write("{}: {}".format(character, tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
