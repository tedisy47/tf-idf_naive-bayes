from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd
import numpy as np
import math

import json
import tfidf.naive_bayes as nb






def tokenize(filename):
    filename = filename
    df = pd.read_csv('media/'+filename)
    df['komentar'] = df.groupby(['sentimen'])['komentar'].transform(lambda x : ' '.join(x))
    df = df.drop_duplicates()

    token = []
    
    for index, row in df.iterrows():
        # print(row['komentar'], row['sentimen'])
        # print(index)
        kalimat = steming(row['komentar'])
        # print()
        kata = kalimat.split(' ')
        kata = np.array(kata)
        for kataToken in kata:
            if kataToken not in token:
                token.append(kataToken)
        # print(kata)
    hasil = tfIdf(token,filename)
    return hasil

def tfIdf(token,filename):

    df = pd.read_csv('media/'+filename)
    df['komentar'] = df.groupby(['sentimen'])['komentar'].transform(lambda x : ' '.join(x))
    df = df.drop_duplicates()
    data = {}
    n = 0
    doc = []
    doc_all = {}
    key_doc = []
    for index, row in df.iterrows():
        n = n+1
        # print(row['komentar'], row['sentimen'])
        # print(index)
        kalimat = steming(row['komentar'])
        kata = kalimat.split(' ')
        doc.append(kata)
        computeTF = computetf(kata,token)
        # print(computeTF);
        document = index +1
        data[row['sentimen']] = computeTF
        key_doc.append(row['sentimen'])
    # print(token)
    hasil = pd.DataFrame(data, token)
    hasil.to_csv('hasil.csv',index=True)

    with open('key_doc.json', 'w') as f:
        json.dump(key_doc, f)
    with open('filename.json', 'w') as f:
        json.dump(filename, f)
    return hasil
    
   
def perior(hasil):
    # print(key_doc)
    with open('key_doc.json') as f_in:
        key_doc = json.load(f_in)

    with open('filename.json') as f_in:
        filename = json.load(f_in)

   
    perior = nb.perior(hasil,key_doc,filename)
    return perior

def conditional(hasil):
    
    with open('key_doc.json') as f_in:
        key_doc = json.load(f_in)
    conditional = nb.conditional(hasil,key_doc)
    smooting = nb.laplace_smoothing(conditional,key_doc)
    csv_datasest = nb.create_csv(smooting,key_doc)
    
    return smooting

    
# menghitung tf.idf
def comupte_tf_idf(key,hasil,key_doc):
    # print(document)
    for x in key_doc:
        hasil['tfidf_'+x] = hasil[x] * hasil['idf']

    return(hasil)



# menghitung df dan idf
def computedf(token,doc,n):
    df_hasil = []
    idf = []
    list_hasil = {}
    for x in token:
        hasil = 0
        for row in doc:
            if x in row:
                hasil = hasil + 1
            else:
                hasil= hasil
        df_hasil.append(hasil)
        log_cuy = n/hasil
        hasi_idf = math.log10(log_cuy)
        idf.append(hasi_idf)
    list_hasil['df'] = df_hasil
    list_hasil['idf'] = idf
    return list_hasil



    # menghitung TF
def computetf(kata,token):
    hasil = {}
    for x in token:
        count = 0
        for y in kata:
            
            if x == y:
                count +=1
        hasil[x] = count
    return hasil

      
    # print(tfDict)
            



def steming(sentence=None):
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    output   = stemmer.stem(sentence)
    return output
    

