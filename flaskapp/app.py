import numpy as np
import uuid
from flask import Flask, request, make_response
from flask_cors import cross_origin
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime

# Use the application default credentials
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
  'projectId': project_id, #replace with your project-id
})
db = firestore.client()

app = Flask(__name__)

session_id=uuid.uuid4().hex
tokenizer = AutoTokenizer.from_pretrained('./model_n4')
model =  AutoModelForCausalLM.from_pretrained('./model_n4')
prompt_ids = tokenizer.encode('jon snow'+':', return_tensors='pt') # can be changed to another character e.g. arya stark

# getting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():
    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

# processing the request from dialogflow
def processRequest(req):
    result = req.get("queryResult")

    #Fetching the data points
    convo_id = req.get("responseId")
    new_user_input=result.get("queryText")
    new_user_input_ids = tokenizer.encode(new_user_input + tokenizer.eos_token, return_tensors='pt')

    doc_ref = db.collection(session_id)
    docs = doc_ref.get()
    #Retrieving chat history if exists
    if docs:
        #get chat history from firebase
        query = doc_ref.order_by(u'time_stamp', direction=firestore.Query.DESCENDING).limit(1)
        doc = [item for item in query.stream()][0]
        chat_history_ids = tokenizer.encode(doc.to_dict()['chat_history'],return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids, prompt_ids], dim=-1)
        bot_input_ids = bot_input_ids[:np.min([bot_input_ids.size()[-1],1024])]
    else:
        bot_input_ids = torch.cat([new_user_input_ids, prompt_ids], dim=-1)
        bot_input_ids = bot_input_ids[:np.min([bot_input_ids.size()[-1],1024])]

    #Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')

    #Fitting out model with the data points
    if (intent=='UserInput'):
        chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
        )

        chat_history = tokenizer.decode(chat_history_ids[0])
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        doc_ref = db.collection(session_id).document(convo_id)
        doc_ref.set({
            u'chat_history': chat_history,
            u'time_stamp': datetime.now()
        })


        #Returning back the fullfilment text back to DialogFlow
        return {
            "fulfillmentText": output
        }


if __name__ == '__main__':
    app.run()
