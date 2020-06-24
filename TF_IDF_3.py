import pandas as pd
import array
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from pip._vendor.distlib.compat import raw_input


df = pd.read_csv('Amazon_reviews.csv')
saved_column = df.Review
print(saved_column)
########################################################

def count_occurence(word,sentence):
    return sentence.count(word)
##########################################################
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
#############################################################################
def  NormalizedPositive(numpy2):
    #print(numpy2)
    numpy1 = np.array([])
    for a,b,c,d,e in numpy2:
        sum = 0
        sum = math.sqrt((a*a) + (b*b) + (c*c) +(d*d)+(e*e))
        if(sum != 0):
            a = a/sum
            b = b/sum
            c = c/sum
            d = d/sum
            e  =e/sum
        else:
            a =0
            b =0
            c =0
            d =0
            e =0
        numpy1 = np.append(numpy1,[a,b,c,d,e])
    numpy1 = numpy1.reshape(199,5)
    print("Normalized matrix")
    #print(numpy1)
    return numpy1


###########################################################################
def positive_index():
    list =['stuning','amazing','best','happy','good']
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
                #temp['logtf'] = 1+ am
            if(temp['df']!=0):
                temp['idf'] = math.log10(199/temp['df' ])
            else:
                temp['idf'] = 0
            temp['TF-idf'] = temp['log-tf']*temp['idf']


            #arr.append(temp['tf'])
            numpy1 = np.append(numpy1, temp['tf'])
            numpy2 = np.append(numpy2, temp['TF-idf'])
            #print("term frequency of "+name)

    numpy1=numpy1.reshape(199,5)
    numpy2=numpy2.reshape(199,5)
    #calculat = NormalizedPositive(numpy2)
    print("The count matrix of the above is : COLUMNS ARE THE POSITIVE AND NEGATIVE WORST AND ")
    print("COLUMNS ARE THE POSITIVE AND NEGATIVE WORSDS")
    print("ROWS ARE THE REVIEWS ")
    #print(numpy1)
    print("tf-idf Matrix")
    #print(numpy2)
    return(numpy2)


####################################################################################################
def negative_index():
    list = [ 'poor', 'disappointed', 'bad', 'beware', 'waste']
    numpy1 = np.array([])
    numpy2 = np.array([])
    arr = []
    for row in saved_column:
        i = 0
        counts = {}
        temp = {}

        row = re.sub('[^A-Za-z0-9]+', ' ', row)
        row = row.lower()

        for list[i] in list:
            # print(row)
            name = list[i]
            counts[name] = temp
            temp['tf'] = count_occurence(name, row)
            check = document_frequency()
            temp['df'] = check[name]
            varia1 = temp['tf']
            # am = math.log10(varia1)
            if (temp['tf'] != 0):
                temp['log-tf'] = 1 + (math.log10(temp['tf']))
            else:
                temp['log-tf'] = 0
                # temp['logtf'] = 1+ am
            if (temp['df' ]!= 0):
                temp['idf'] = math.log10(199 / temp['df'])
            else:
                temp['idf'] = 0
            temp['TF-idf'] = temp['log-tf'] * temp['idf']

            # arr.append(temp['tf'])
            numpy1 = np.append(numpy1, temp['tf'])
            numpy2 = np.append(numpy2, temp['TF-idf'])
            # print("term frequency of "+name)

    numpy1 = numpy1.reshape(199, 5)
    numpy2 = numpy2.reshape(199, 5)
    print("The count matrix of the above is : COLUMNS ARE THE POSITIVE AND NEGATIVE WORST AND ")
    print("COLUMNS ARE THE POSITIVE AND NEGATIVE WORSDS")
    print("ROWS ARE THE REVIEWS ")
    #print(numpy1)
    print("tf-idf Matrix")
    #print(numpy2)
    return (numpy2)


#EUCLIDEAN DISTANCES
def euclidean_distance(x1, x2):

    return np.sqrt(np.sum((x1 - x2) ** 2))


# KMEANS FUNCTIONS
def mykmeans(X,k,c,count):
   count = count +1
   x =X
   #print(x)

   Kluster = k
   #center1
   cen1 = c[0]
   x01 = cen1[0]
   y01 = cen1[1]
   center1 = np.array([x01,y01])
   #center 2
   cen2 = c[1]
   x02 = cen2[0]
   y02 = cen2[1]
   center2 = np.array([x02, y02])
   #print(center2)

   #CLUSTER !
   OutputData1 = np.zeros(2)
   #print(type(OutputData1))

   #CLUSTER2
   OutputData2 = np.zeros(2)
   #print(X)


   #THE CLUSTER, THE POINT BELONGS TO
   for row in  X:

       x1 = row[0]
       y1 = row[1]
       i = np.array([x1,y1])


       distance1 = euclidean_distance(i,center1)

       distance2 = euclidean_distance(i,center2)

       #if distance of closest cluster

       if(distance1 < distance2):
           OutputData1 =  np.append(OutputData1,[x1,y1])
           si = (OutputData1.size)/2
           so = int(si)

           OutputData1 = OutputData1.reshape(so,2)

       else:
           OutputData2 = np.append(OutputData2, [x1, y1])
           ji = (OutputData2.size) / 2
           jo = int(ji)

           OutputData2 = OutputData2.reshape(jo, 2)




   N1 = (OutputData1.size)/2
   N2 = (OutputData2.size)/2


   Output_data_center = np.zeros(2)
   Output_data_center2 = np.zeros(2)
   a = np.mean(OutputData1,axis=0)
   b = np.mean(OutputData2,axis=0 )


   final_center = np.array([])
   final_center = np.append(final_center,a)
   final_center = np.append(final_center, b)
   final_center = final_center.reshape(2,2)


   # compute the new centroid of output data1

   #difference of new and old center
   diff = euclidean_distance(a,center1)
   #print("difference 1")
   #print(diff)
   diff2 = euclidean_distance(b,center2)
   #print("difference 2")
   #print(diff2)
   if(diff<0.001 and diff2 <0.001):
       print("the final number of iterations taken is")
       print(count)
       pass

   else:
       X1 = np.concatenate((OutputData1, OutputData2), axis=0)
       mykmeans(X1, 2, final_center, count)

   return (OutputData1, OutputData2, final_center, count)

########################################################################################
def main():
 a = positive_index()
 b = NormalizedPositive(a)
 #sum of all positive values in a row
 x = np.sum(b,axis=1)
 x = x.reshape(199,1)

####################################
 c = negative_index()
 d = NormalizedPositive(c)
 # sum of all negatives in values in a row
 y = np.sum(c, axis=1)
 y = y.reshape(199, 1)
 #append x and y
 X1 = np.concatenate((x, y), axis=1)

 #print(X1)
#######################################################################################################

 c1 = [5, 1.1527]
 c2 = [0, 5.0903]
 c = []
 c.append(c1)
 c.append(c2)
 count = 0

 # CALLING THE KMEANS FUNCTION
 numpy1 = np.array([])
 numpy2 = np.array([])
 numpy3 = np.array([])
 int1 = int()
 numpy1, numpy2, numpy3, count = mykmeans(X1, 2, c, count)
 print("the final center is ")
 print(numpy3)
 for x, y in numpy1:
     plt.plot(x, y, marker="o", color="blue", markeredgecolor="black")
 #plt.scatter(numpy3[0],  marker="x", color='b')
 for x, y in numpy2:
     plt.plot(x, y, marker="o", color="red", markeredgecolor="black")
 for x, y in numpy3:
     plt.plot(x, y, marker="x", color="black", markeredgecolor="black")

 plt.show()
 exit()


if __name__ == '__main__':
    main()
