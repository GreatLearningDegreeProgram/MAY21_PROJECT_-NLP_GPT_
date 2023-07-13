from transformers import pipeline
from spacy import displacy
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from config import NER_COLORS


cmap = plt.cm.get_cmap('rainbow', len(NER_COLORS))
ner_colors = {k:matplotlib.colors.rgb2hex(cmap(v-1)) for k,v in NER_COLORS.items()}
ner_1 = pipeline("ner", grouped_entities=True)
ner_2 = pipeline("ner", model="mrm8488/mobilebert-finetuned-pos", grouped_entities=True)

def convert_hf_to_displacy_format(hf_pred, _original_text, _title=None):
    """ Function to convert prediction to the displacy specific format """
    return [dict(
        text=_original_text, 
        ents=[{
            "start":ent["start"], 
            "end":ent["end"], 
            "label":ent["entity_group"], 
            "score":ent["score"]} for ent in hf_pred], 
        title=_title
    ),]



def getDispacy(text):
    original_text = text    
    ner_pred = ner_1(original_text)
    return displacy.render(convert_hf_to_displacy_format(ner_pred, original_text), style="ent", manual=True,jupyter=False)
