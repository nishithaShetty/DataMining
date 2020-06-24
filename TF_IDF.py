import pandas as pd
import array
import re
import numpy as np
import math
df = pd.read_csv('Amazon_reviews.csv')
saved_column = df.Review
print(saved_column)
########################################################


def document_frequency():
    dfcounter = {}
    for row in saved_column:
        counter = {}
        counts = {}
        temp = {}
        lst = []
        data = {}

        words = row.split()
        for word in words:
            word = re.sub('[^A-Za-z0-9]+', ' ', word)
            word = word.lower()
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
for row in saved_column:
     words = row.split()
dfcounter={}
numpy1 = np.array([])
numpy2 = np.array([])
i= 0
for row in saved_column:
    counter = {}
    counts = {}
    temp = {}
    lst = []
    data = {}


    words =row.split()
    for word in words:
        word = re.sub('[^A-Za-z0-9]+', ' ', word)
        word = word.lower()
        if word not in counts:

                temp['tf-'+word] = 1
                temp['log-tf-'+word] = 1 + (math.log10(temp['tf-'+word]))
                check = document_frequency()
                temp['df-'+word] = check[word]

                temp['idf-'+word] = math.log10(199 / temp['df-'+word])

                temp['TF-idf-'+word] = temp['log-tf-'+word] * temp['idf-'+word]

                counts[word] = temp

                numpy1=np.append(numpy1,temp['tf-'+word])
                numpy2 = np.append(numpy2, temp['TF-idf-'+word])


            #lst.append(word)
        else:
                temp = counts.get(word)
                data=temp.get('tf-'+word)+1
                temp['tf-'+word] = data
                if (temp['tf-'+word] != 0):
                    temp['log-tf-'+word] = 1 + (math.log10(temp['tf-'+word]))
                else:
                    temp['log-tf'] = 0


                check = document_frequency()
                temp['df-'+word ] = check[word]

                temp['idf-'+word] = math.log10(199 / temp['df-'+word])

                temp['TF-idf-'+word] = temp['log-tf-'+word ] * temp['idf-'+word]
                numpy1 = np.append(numpy1, temp['tf-'+word])
                numpy2 = np.append(numpy2, temp['TF-idf-'+word ])
    i = i + 1
    print("Data for ROW:" )
    print(i)
    print("the term frequency is :")
    print(numpy1)
    print("the TF/IDF is :")
    print(numpy2)


#print("the document frequency is :")
#print(counts['the']['df-the'])
    #print(counts.keys())



