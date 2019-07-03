from flask import Flask
from pymongo import MongoClient
from bson import json_util
from bson.dbref import DBRef
from bson.objectid import ObjectId
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import spell
from tensorflow import keras

import tensorflow as tf

import pandas as pd
import csv
import re
import json


MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DB_NAME = 'mitdev'
COLLECTION_NAME = 'incident'
COLLECTION_NAME_LANDSCAPE='landscape'

data = []
lableData=[]

def index():
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = connection[DB_NAME][COLLECTION_NAME]
    collection_landscape = connection[DB_NAME][COLLECTION_NAME_LANDSCAPE]
    projects = collection.find({'is_active':1})
    json_projects = []
    complete_data=[]
    for project in projects:
        json_projects.append(project['short_description'])
        for landscapelist in project['affected_landScape_id']:
            landscape= collection_landscape.find({'_id': ObjectId(landscapelist.id)})
            lableData.append(convertLableToNumber(landscape.next()['name']))
            # print(preprocess(project['short_description']))
            data.append(preprocess(project['short_description']))

    dict={'label':lableData, 'data':data}
    df=pd.DataFrame(dict)
    df.to_csv('../data/data1.csv')
    #saveDataInCSV(data,lableData)
    json_projects = json.dumps(json_projects, default=json_util.default)
    print(len(data))
    print(len(lableData))
    connection.close()
    train_model(data,lableData)
    return json_projects

def mainfnc():
    df = pd.DataFrame(dict)
    df.to_csv('../data/data1.csv')
    # saveDataInCSV(data,lableData)
    json_projects = json.dumps(json_projects, default=json_util.default)
    print(len(data))
    print(len(lableData))
    connection.close()
    train_model(data, lableData)
    return json_projects

def convertLableToNumber(label):

    switcher = {
        "AGS Support Infras Premium Systems (VH)":1,
        "Artifactory (Internal)":2,
        "DITA":3,
        "GIT" :4,
        "GLDS - Other" : 5,
       "HANA Build": 6,
        "IBSO_INT": 7,
        "Jenkins as a Service":8,
        "Perforce":9,
        "TDC - Other":10,
       "TDC Demo Showroom (CDA)": 11,
        "TDC Demo Showroom (Target/Productive) AMER":12,
        "TDC Demo Showroom (Target/Productive) APJ":13,
        "TDC Demo Showroom (Target/Productive) EMEA":14,
        "TDC Education Partner Landscape (Target/Productive)":15,
        "TDC Live Access (Target/Productive)":16,
        "TDC Training Landscape (Target/Productive)":17
    }
    return switcher.get(label, 0)


def saveDataInCSV(data,label):
    print(data)
    print(label)
    with open('../data/data.csv', 'w') as writeFile:
        writer_data = csv.writer(writeFile)
        for rowdata in data:
            writer_data.writerows(rowdata)
    with open('../data/label.csv', 'w') as writeFile:
        writer_label = csv.writer(writeFile)
        writer_label.writerows(label)

def preprocess(data):
    data=text_cleaner(data)
    word_token=word_tokenize(data)
    stop_words = set(stopwords.words('english'))
    filteredWords=[spell(w) for w in word_token if not w in stop_words]
    #print(filteredWords)
    return ' '.join(filteredWords)

def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'}, # remove spaces after a tag opens or closes
        {r'\s+': u' '}, # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'}, # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'}, # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'}, # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''}, # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'}, # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''}, # remove remaining tags
        {r'^\s+': u''} # remove spaces at the beginning
        ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
            text = text.rstrip()
    return text.lower()


def train_model(data,label):
    x_val = data[:75]
    partial_x_train = data[75:]
    y_val = label[:75]
    partial_y_train = label[75:]
    train_tf_model(x_val,y_val,partial_x_train,partial_y_train)

def train_tf_model(train_data,train_label,test_data,test_label):
    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_data,
                        train_label,
                        epochs=40,
                        batch_size=512,
                        validation_data=(test_data, test_label),
                        verbose=1)

    results = model.evaluate(test_data, test_label)
    print(results)



index()