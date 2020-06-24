import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import math

#creating GAUSSIANS DATASET
from pip._vendor.distlib.compat import raw_input



def sampleSet(N1,N2):
    mean1 = [1, 0]
    mean2 = [0 , 1]
    covar1 = [[1, 0.75], [0.75, 0.1]]
    covar2 = [[1, 0.75], [0.75, 0.1]]
    set1 = np.random.multivariate_normal(mean1, covar1, N1)
    set2 = np.random.multivariate_normal(mean2, covar1, N2)
    return set1,set2

def predict(x,m0,v0,m1,v1,p0,p1):
    _pdf0 =np.array([])
    _pdf1 =np.array([])

    prior_c0 = p0

    prior_c1 = p1



    _pdf0 = np.append(_pdf0,(_pdf(x[0], m0[0], v0[0])))
    _pdf0 = np.append(_pdf0,(_pdf(x[1], m0[1], v0[1])))

    _pdf1 = np.append(_pdf1,(_pdf(x[0], m1[0], v1[0])))
    _pdf1 = np.append(_pdf1,(_pdf(x[1], m1[1], v1[1])))
    posterior0 = (prior_c0 * np.prod(_pdf0))

    posterior1 = (prior_c1 * np.prod(_pdf1))


    if(posterior0 > posterior1):
        return 0
    elif(posterior0 <posterior1):
        return 1


# Calculate the Gaussian probability distribution function for x
def _pdf(x, mean, stdev):
	exponent = np.exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent



def accuracy(count,num_samples):
    abc =count/num_samples
    return abc



def myNB(X, Y, Xtest, Ytest):
     training_set = np.concatenate((X,Y),axis=1)
     training_size = (training_set.size)

     ##### class0_training
     ##### class1_training

     class0_training =np.array([])
     class1_training =np.array([])
     class0_mean = np.array([])
     class1_mean = np.array([])
     class0_var = np.array([])
     class1_var = np.array([])

     for x in training_set:
         if(x[2]== 0.0):

             class0_training = np.append(class0_training,(x[0],x[1]))
         else:

             class1_training = np.append(class1_training,(x[0],x[1]))
     class0_size =int(class0_training.size/2)
     class0_training =class0_training.reshape(class0_size,2)

     class0_mean =np.mean(class0_training,axis=0)
     class0_var = np.var(class0_training, axis=0)
     class0_SD0 = np.sqrt(class0_var)
     prior_class0 = class0_size/training_size


#####################################################
     class1_size = int(class1_training.size / 2)
     class1_training = class1_training.reshape(class1_size, 2)

     class1_mean = np.mean(class1_training, axis=0)
     class1_var = np.var(class1_training, axis=0)
     class1_SD1 = np.sqrt(class1_var)
     prior_class1 = class1_size / training_size


     predictions = np.array([])
     for x in Xtest:
         o = predict(x,class0_mean,class0_SD0,class1_mean,class1_SD1,prior_class0,prior_class1)
         predictions = np.append(predictions, o)
     size_table= int(predictions.size)
     predictions = predictions.reshape(size_table,1)

     return predictions



def main():
    accuracy_list = np.array([])
    N1 = raw_input("Enter Sample size for class 0  : ")
    N1 = int(N1)
    N2 = raw_input("Enter Sample size for class 1  : ")
    N2 = int(N2)

    for it in range(1, 10):
        b = sampleSet(N1,N2)
        print("First sample set")
        ## size of training dataset having class 0
        class0_size = b[0].size/2
        ## creating an array with class 0 - 500 training dataset with class0
        class0 = np.array([])
        class0 =np.repeat(0,class0_size)
        class0_size = int(class0_size)
        class0 = class0.reshape(class0_size,1)

        # array with the points x,y and the class corresponding to it
        class0_full =np.array([])
        class0_full = np.concatenate((b[0],class0),axis=1)



        print("Second sample set")
        ## size of training dataset having class 0
        class1_size = b[1].size / 2
        ## creating an array with class 0 - 500 training dataset with class0
        class1 = np.array([])
        class1 = np.repeat(1, class1_size)
        class1_size = int(class1_size)
        class1 = class1.reshape(class1_size, 1)

        # array with the points x,y and the class corresponding to it
        class1_full = np.array([])
        class1_full = np.concatenate((b[1], class1), axis=1)
        #print(class1_full)

        #############################
        #TEST DATA
        c = sampleSet(N1,N2)
        print("First TEST sample set")
        ## size of training dataset having class 0
        class0_size_test = c[0].size / 2
        ## creating an array with class 0 - 500 training dataset with class0
        class0_test = np.array([])
        class0_test = np.repeat(0, class0_size_test)
        class0_size_test = int(class0_size)
        class0_test = class0_test.reshape(class0_size_test, 1)

        # array with the points x,y and the class corresponding to it
        class0_full_test = np.array([])
        class0_full_test = np.concatenate((b[0], class0), axis=1)
        #print(class0_full_test)
        #print (class0_full_test.size)
        # second  TEST Dataset###########

        print("Second sample set")
        ## size of training dataset having class 0
        class1_size_test = c[1].size / 2
        ## creating an array with class 0 - 500 training dataset with class0
        class1_test = np.array([])
        class1_test = np.repeat(1, class1_size_test)
        class1_size_test = int(class1_size_test)
        class1_test = class1_test.reshape(class1_size_test, 1)

        # array with the points x,y and the class corresponding to it
        class1_full_test = np.array([])
        class1_full_test = np.concatenate((c[1], class1_test), axis=1)
        #print(class1_full_test)
        #print (class1_full_test.size)



    ##########################################################
        #sending the concatenated data
        X = np.array([])
        Y = np.array([])
        X = np.concatenate((b[0],b[1]), axis=0)
        Y = np.concatenate((class0,class1),axis=0)
        Xtest = np.array([])
        Ytest = np.array([])
        Xtest = np.concatenate((c[0], c[1]), axis=0)

        Ytest = np.concatenate((class0_test, class1_test), axis=0)
        Xfull = np.concatenate((Xtest, Ytest), axis=1)
        print("Before  prediction")
        for x, y, z in Xfull:
            if (z == 0):
                plt.plot(x, y, marker="o", color="blue", markeredgecolor="black")
            else:
                plt.plot(x, y, marker="o", color="red", markeredgecolor="black")
        plt.show()
        print("After  prediction")
        predictions=myNB(X, Y, Xtest, Ytest)
        Xfull1 = np.concatenate((Xtest, predictions), axis=1)

        for x, y, z in Xfull1:
            if (z == 0):
                plt.plot(x, y, marker="o", color="blue", markeredgecolor="black")
            else:
                plt.plot(x, y, marker="o", color="red", markeredgecolor="black")
        plt.show()

        True_Positives = 0
        True_Negatives = 0
        False_Positives =0
        False_Negatives = 0
        confusion_matrix =np.array([])
        for i, j in np.nditer([predictions, Ytest]):
            if (i == j):

                if(i == 0):
                    True_Positives = True_Positives + 1
                else:
                    True_Negatives =True_Negatives + 1
            else:
                if (i == 0):
                    False_Positives = False_Positives + 1
                else:
                    False_Negatives = False_Negatives + 1

        print("The confusion matrix for iteration ")
        print(it)
        confusion_matrix = np.append(confusion_matrix, True_Positives)
        confusion_matrix = np.append(confusion_matrix, False_Negatives)
        confusion_matrix = np.append(confusion_matrix, False_Positives)
        confusion_matrix = np.append(confusion_matrix, True_Negatives)
        confusion_matrix = confusion_matrix.reshape(2, 2)
        print (confusion_matrix)


        count =0
        count1 =0


        for i, j in np.nditer([predictions, Ytest]):
                    if (i == j):

                            count = count + 1
                    else:

                            count1 = count1 + 1

        acc = accuracy(count,Ytest.size)
        print(acc)
        accuracy_list =np.append(accuracy_list,acc)
    print ("the precision is:")
    print(accuracy_list)
    print("The average is")
    print (np.mean(accuracy_list))



if __name__ == '__main__':
    main()


