# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:56:34 2024

@author: pathouli
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def predict_next_word(text):
    # Encode the input text
    inputs = tokenizer.encode(text, return_tensors='pt')

    # Generate predictions
    outputs = model.generate(
        inputs, max_length=len(inputs[0]) + 1, do_sample=False)

    # Decode the output and extract the predicted next word
    predicted_text = tokenizer.decode(
        outputs[0], skip_special_tokens=True)
    next_word = predicted_text[len(text):].split()[0]
    
    return next_word

text = "The bear chased the deer down the"
for i in range(0, 10):
# Example usage
    next_word = predict_next_word(text)
    text = text + " " + next_word
    #print(text)
print (text)
