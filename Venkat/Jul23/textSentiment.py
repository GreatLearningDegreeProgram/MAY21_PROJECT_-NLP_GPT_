from textblob import TextBlob
from transformers import pipeline
import streamlit as st
from config import CLASSIFIER_LABEL

classifier = pipeline("zero-shot-classification")


def pretty_print_zero_shot( _sequences):
    data = {}
    retvalues =  classifier(sequences=_sequences,candidate_labels=CLASSIFIER_LABEL).items()
    for k,v in retvalues: 
        data[k] = v
    li = data['scores']
    maxscore = max(li)
    maxindex = li.index(max(li))
    
    return data['labels'][maxindex]
    
def getLable():
    return CLASSIFIER_LABEL   


def getSentiment(text):
    
    my_sentence = TextBlob(text)
    return my_sentence.sentiment
