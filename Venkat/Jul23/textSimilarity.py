import transformers
import numpy 

# Load the BERT model

model_class,tokenizer_class,pretrained_weights = transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def getSimilarityScore(text1,text2):
    encoding1 = tokenizer.encode(text1, add_special_tokens=True, max_length=512) 
    encoding2 = tokenizer.encode(text2, add_special_tokens=True, max_length=512)
    similarity_score = numpy.dot(encoding1, encoding2) / (numpy.linalg.norm(encoding1) * numpy.linalg.norm(encoding2))
    return round(similarity_score,2)