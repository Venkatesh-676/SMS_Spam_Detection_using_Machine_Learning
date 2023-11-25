from asyncio.windows_events import NULL
from django.shortcuts import render
from django.template import Context, Template

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import nltk
tfidf = pickle.load(open('D:\model04.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    import string
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

def prediction(request):
    if request.method=='POST':
        s=""
        d=request.POST
        for key,value in d.items():
            if key=="SMS":
                s=value
        transformed_sms = transform_text(s)
        vector_input = tfidf.transform([transformed_sms])
        loaded_model = pickle.load(open("D:\model10.pkl", "rb")) 
        result = loaded_model.predict(vector_input)[0]
        print(result)

        if result==1:
            context = {"prediction": "Spam"}
            return render(request,"result.html",context)
        else:
            context = {"prediction": "Not Spam"}
            return render(request,"result.html",context)
    return render(request,'main.html')
