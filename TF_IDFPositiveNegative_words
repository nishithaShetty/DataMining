import pandas as pd
import array
import numpy as np
import re
import math
df = pd.read_csv('Amazon_reviews.csv')
saved_column = df.Review
print(saved_column)
########################################################

def count_occurence(word,sentence):
    return sentence.count(word)



def document_frequency():
    dfcounter = {}
    for row in saved_column:
        counter = {}
        counts = {}
        temp = {}
        lst = []
        data = {}
        row = re.sub('[^A-Za-z0-9]+', ' ', row)
        row = row.lower()
        words = row.split()
        for word in words:

            if word not in counts:
                counts[word] = 1
                if word in dfcounter:

                    dfcounter[word] = dfcounter.get(word) + 1
                else:

                    dfcounter[word] = 1
                # lst.append(word)
            else:
                data = counts.get(word) + 1
                counts[word] = data


    return (dfcounter)


###########################################################################
list =['stuning','amazing','best','happy','great','poor','disappointed','bad','beware','waste']
numpy1 = np.array([])
numpy2 = np.array([])
arr=[]
for row in saved_column:
    i = 0
    counts = {}
    temp ={}

    row = re.sub('[^A-Za-z0-9]+', ' ', row)
    row = row.lower()

    for list[i] in list:
        #print(row)
        name = list[i]
        counts[name]=temp
        temp['tf'] = count_occurence(name,row)
        check = document_frequency()
        temp['df' ] = check[name]

        varia1 = temp['tf']
        #am = math.log10(varia1)
        if(temp['tf']!=0):
            temp['log-tf'] = 1 +(math.log10(temp['tf']))
        else:
            temp['log-tf'] = 0

        if(temp['df']!=0):
            temp['idf'] = math.log10(199/temp['df' ])
        else:
            temp['idf'] = 0
        temp['TF-idf'] = temp['log-tf']*temp['idf']


        #arr.append(temp['tf'])
        numpy1 = np.append(numpy1, temp['tf'])
        numpy2 = np.append(numpy2, temp['TF-idf'])
        #print("term frequency of "+name)

numpy1=numpy1.reshape(199,10)
numpy2=numpy2.reshape(199,10)
print("The count matrix of the above is : COLUMNS ARE THE POSITIVE AND NEGATIVE WORST AND ")
print("COLUMNS ARE THE POSITIVE AND NEGATIVE WORSDS")
print("ROWS ARE THE REVIEWS ")
print(numpy1)
print("tf-idf Matrix")
print(numpy2)


