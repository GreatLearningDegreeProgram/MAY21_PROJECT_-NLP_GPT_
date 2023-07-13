import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('all')

def getcosineSimilarity(text1,text2):
    x_list = word_tokenize(text1.lower())
    y_list = word_tokenize(text2.lower())

    sw = stopwords.words('english') 
    t1 =[]
    t2 =[]

    x_set = {word  for word in x_list if not word in sw }
    y_set = {word  for word in y_list if not word in sw }
    rvector = x_set.union(y_set) 

    for wrd in rvector:
        if wrd in x_set:  t1.append(1) 
        else: t1.append(0)
        if wrd in y_set: t2.append(1)
        else: t2.append(0)

    c = 0
    for i in range(len(rvector)):
        c+= t1[i]*t2[i]
        cosineScore = c / float((sum(t1)*sum(t2))**0.5)
    return cosineScore

def getjaccardSimilarity(text1, text2):
    x_list = word_tokenize(text1.lower())
    y_list = word_tokenize(text2.lower())

    intersection = len(list(set(x_list).intersection(y_list)))
    union = (len(x_list) + len(y_list)) - intersection
    return round(float(intersection) / union,2)