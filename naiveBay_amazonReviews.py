import pandas as pd
import re
import numpy as np
import math

def read_file():
    df = pd.read_csv('Amazon_reviews.csv')
    saved_column = df.Review
    saved_column2 = df.Label
    lst =[]
    thisdict ={}
    k=[]
    word =[]
    f = open("abc.txt", "r")
    for wordss in f:
        wordd = wordss.split(',')

    for row in saved_column:
         words = row.split()
         row = re.sub('[^A-Za-z]+', ' ', row)
         row = row.lower()
         for word in words:
             word = re.sub('[^A-Za-z]+', ' ', word)
             word = word.lower()
             word = word.replace(" ", "")

             if word not in wordd:
                  if word not in lst:
                      lst.append(word)
    return(df,saved_column,saved_column2,lst)

################################################################################################
######DATA preProcessing


def document_frequency(saved_column,lst):
    counts1 = {}
    document_f = np.array([])
    idf = np.array([])
    count =0
    for row in saved_column:
        lst2 = []
        words = row.split()
        for wordz in words:

            wordz = re.sub('[^A-Za-z]+', ' ', wordz)
            wordz = wordz.lower()
            wordz = wordz.replace(" ", "")

            lst2.append(wordz)


        for word in lst:
            if word in lst2:
                if word in counts1:
                        data = counts1.get(word)
                        counts1[word] = data + 1
                else:
                        counts1[word] =1


    print(counts1.get("soundtrack"))
    for i in lst:
        document_f = np.append(document_f,counts1.get(i))
        idf_value= math.log10(50 / counts1.get(i))
        idf = np.append(idf, idf_value)

    return document_f,idf

###########################################################################
###########TERM FREQUENCY

def checkK(i,counts):
    if(i in counts.keys()):
        return 0
    else:
        return 1




def term_frequency(saved_column,lst):
    term_f= np.array([])
    cou = 0
    log_tf =np.array([])
    for row in saved_column:
        counter = {}
        counts = {}
        temp = {}

        data = {}
        lst2 =[]
        words =row.split()
        for wordz in words:

            wordz = re.sub('[^A-Za-z]+', ' ', wordz)
            wordz = wordz.lower()
            wordz = wordz.replace(" ", "")

            lst2.append(wordz)


        for word in lst2:
            if word in lst:

                if word not in counts:
                    counts[word] = 1

                    #check = document_frequency(saved_column,lst)
                    #lst.append(word)
                else:

                    data = counts.get(word) + 1
                    counts[word] = data



        for i in lst:
          ret = checkK(i,counts)
          if(ret == 0):
            term_f = np.append(term_f,counts.get(i))
            asd = 1+ (math.log10(counts.get(i)))
            log_tf = np.append(log_tf, asd)

          else:
            term_f = np.append(term_f, 0)
            log_tf =np.append(log_tf,0)


    term_f = term_f.reshape(50,len(lst))
    log_tf = log_tf.reshape(50, len(lst))

    return term_f,log_tf



    #print("Document Frequency")
    #print(dfcounter)

def TF_IDF(log_tf,idf):
    tf_idf = np.array([])
    col = 0

    for i in np.nditer(idf):
        log_tf[:, col] *= i
        col = col + 1
    # print(log_tf)
    return log_tf
def Normalize(tf_idf):
    norma =np.array([])
    tf_idf =np.square(tf_idf)
    norma = np.sum(tf_idf,axis=1)
    norma = np.sqrt(norma)
    return norma



###########################################################################
#########NAIVE BAYES

def predict(x,m0,v0,m1,v1,p0,p1):


    _pdf0 =np.array([])
    _pdf1 =np.array([])

    prior_c0 = np.log(p0)
    prior_c1 = np.log(p1)

    for i, j,k in np.nditer([x, m0,v0]):
           if(k==0):
               pass
           else:
             add = np.log(pdf(i,j,k))
             _pdf0 = np.append(_pdf0,add)


    for i, j, k in np.nditer([x, m1, v1]):
           if(k==0):
               pass
           else:
            add1 = np.log(pdf(i, j, k))
            _pdf1 = np.append(_pdf1,add1)

    posterior0 = prior_c0 * np.sum(_pdf0)

    posterior1 =  prior_c1 * np.sum(_pdf1)


    if(posterior0 > posterior1):
        return "class1"
    elif(posterior0 < posterior1):
        return "class2"


def pdf(x,mean,std):
    denom = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    num = (1/ (np.sqrt(2 * np.pi) * std)) * denom
    return num




def accuracy(count,num_samples):
    abc =count/num_samples
    return abc



def myNB(X, Y, Xtest, Ytest):
     training_set = np.concatenate((X,Y[:, None]),axis=1)

     training_size = (training_set.size/1361)



     ##### class1_training
     ##### class2_training

     class1_training =np.array([])
     class2_training =np.array([])
     class1_mean = np.array([])
     class2_mean = np.array([])
     class1_var = np.array([])
     class2_var = np.array([])
     count = 0
     count1 = 0

     for x in training_set:
           if (x[1360] == "class1"):
             count = count + 1
             for a in x:
                 if(a=="class1"):
                     pass
                 else:
                     class1_training = np.append(class1_training, float(a))
           else:
             count1 = count1 + 1
             for a in x:
                 if(a=="class2"):
                     pass
                 else:
                    class2_training = np.append(class2_training, float(a))

     class1_size =int(class1_training.size/1360)

     class1_training =class1_training.reshape(class1_size,1360)

     class1_mean =np.mean(class1_training,axis=0)

     class1_var = np.var(class1_training, axis=0)

     class1_SD1 = np.sqrt(class1_var)
     prior_class1 = class1_size/training_size



#####################################################
     class2_size = int(class2_training.size / 1360)
     class2_training = class2_training.reshape(class2_size, 1360)

     class2_mean = np.mean(class2_training, axis=0)

     class2_var = np.var(class2_training, axis=0)

     class2_SD2 = np.sqrt(class2_var)
     prior_class2 = class2_size / training_size



     predictions = np.array([])
     answers =np.array([])
     for x in Xtest:

         o = predict(x,class1_mean,class1_SD1,class2_mean,class2_SD2,prior_class1,prior_class2)
         print(o)
         answers = np.append(answers,o)
         predictions = np.append(predictions, o)
     size_table= int(predictions.size)


     return predictions


def confusion_matri(predictions,test_cla):
    True_Positives = 0
    True_Negatives = 0
    False_Positives = 0
    False_Negatives = 0
    count =0
    confusion_matrix = np.array([])
    for i, j in np.nditer([predictions, test_cla]):
        if (i == j):
            count =count +1

            if (i == "class1"):
                True_Positives = True_Positives + 1
            else:
                True_Negatives = True_Negatives + 1
        else:
            if (i == "class1"):
                False_Positives = False_Positives + 1
            else:
                False_Negatives = False_Negatives + 1

    print("The confusion matrix for iteration ")
    confusion_matrix = np.append(confusion_matrix, True_Positives)
    confusion_matrix = np.append(confusion_matrix, False_Negatives)
    confusion_matrix = np.append(confusion_matrix, False_Positives)
    confusion_matrix = np.append(confusion_matrix, True_Negatives)
    confusion_matrix = confusion_matrix.reshape(2, 2)
    print(confusion_matrix)
    acc =accuracy(count,test_cla.size)
    print(acc)
    return acc


def main():
    a,b,c,d = read_file()
    label= np.array([])
    #print("Term frequency")
    tf,log_tf=term_frequency(b,d)
    print(tf)
    #print("Document frequency")
    df,idf = document_frequency(b,d)
    print("The TF-IDF is")
    tf_idf = TF_IDF(log_tf,idf)
    print(tf_idf)
    print("The size of tf-idf")
    print(tf_idf.size)

    for row in c:
        row = row.replace(" ", "")
        if(row=="__label__2"):
            label = np.append(label,"class2")
        else:
            label = np.append(label,"class1")
    tf_idf_full=np.array([])
    #tf_idf_full=np.concatenate((tf_idf,label[:,None]), axis=1)
    #train_1 = np.split(tf_idf,[10,50][1])
    #####################################################################
    accuracy_list  = np.array([])
    ##########################################################

    ## 1st fold
    test_data = (tf_idf[:10, ])
    train_data = (tf_idf[10:50, ])
    test_cla = (label[0:10])
    train_class = (label[10:50])
    predictions1 = myNB(train_data, train_class, test_data, test_cla)

    ans1 = confusion_matri(predictions1, test_cla)
    print(ans1)
    accuracy_list = np.append(accuracy_list,ans1)
    ##################################################
    ## 2nd fold
    # for 2nd fold
    test_data2 = (tf_idf[10:20, ])
    train_data2 = (tf_idf[20:50, ])
    train_data20 = (tf_idf[0:10, ])
    train_data2_full = np.concatenate((train_data2, train_data20), axis=0)

    test_cla2 = (label[10:20])
    train_class2 = (label[20:50])
    train_class20 = (label[0:10])
    train_class2_full = np.concatenate((train_class2, train_class20), axis=0)
    predictions = myNB(train_data2_full, train_class2_full, test_data2, test_cla2)
    ans2 =confusion_matri(predictions, test_cla2)
    accuracy_list = np.append(accuracy_list, ans2)




    ################################################
    ##3rd fold
    test_data3 = (tf_idf[20:30, ])
    train_data3 = (tf_idf[30:50, ])
    train_data30 = (tf_idf[0:20, ])
    train_data3_full = np.concatenate((train_data3, train_data30), axis=0)

    test_cla3 = (label[20:30])
    train_class3 = (label[30:50])
    train_class30 = (label[0:20])
    train_class3_full = np.concatenate((train_class3, train_class30), axis=0)
    predictions3 = myNB(train_data3_full, train_class3_full, test_data3, test_cla3)
    ans3=confusion_matri(predictions3, test_cla3)

    accuracy_list = np.append(accuracy_list, ans3)
    ###################################################
    # for 4th fold
    test_data4 = (tf_idf[30:40, ])

    train_data4 = (tf_idf[40:50, ])
    train_data40 = (tf_idf[0:30, ])
    train_data4_full = np.concatenate((train_data4, train_data40), axis=0)

    test_cla4 = (label[30:40])
    train_class4 = (label[40:50])
    train_class40 = (label[0:30])
    train_class4_full = np.concatenate((train_class4, train_class40), axis=0)
    predictions4 = myNB(train_data4_full, train_class4_full, test_data4, test_cla4)

    ans4 =confusion_matri(predictions4, test_cla4)

    accuracy_list = np.append(accuracy_list, ans4)

######################################
# fifth fold
    test_data5 = (tf_idf[40:50, ])
    train_data5 = (tf_idf[0:40, ])
    test_cla5 = (label[40:50])
    train_class5 = (label[0:40])
    predictions5 = myNB(train_data5, train_class5, test_data5, test_cla5)

    ans5 = confusion_matri(predictions5, test_cla5)

    accuracy_list = np.append(accuracy_list, ans5)

    print("accuracy list is")
    print(accuracy_list)
    mean_list = np.mean(accuracy_list)
    print("the average is:")
    print(mean_list)







if __name__ == '__main__':
    main()


