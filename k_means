import matplotlib.pyplot as plt
import numpy as np
import sys
import math

#creating GAUSSIANS DATASET
def sampleSet():
    mean1 = [1, 0]
    mean2 = [0,1.5]
    covar1 = [[0.9, 0.4], [0.4, 0.9]]
    covar2 = [[0.9, 0.4], [0.4, 0.9]]
    set1 = np.random.multivariate_normal(mean1, covar1, 500)
    set2 = np.random.multivariate_normal(mean2, covar1, 500)
    return set1,set2

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


def main():
    b =sampleSet()
    #print("First sample set")
    #print(b[0])
    #print("Second sample set")
    #print(b[1])
    b[0].sort()

    #print("mid value is ", b[0][int(len(b[0]) / 2)])
    #print("mid value is ", b[1][int(len(b[1]) / 2)])
    initial_c1 = b[0][int(len(b[0]) / 2)]
    initial_c2 = b[1][int(len(b[1]) / 2)]

    X = np.concatenate([b[0], b[1]])

    print("k-means clustering for k=2")
    #for k=2 clutser computation
    ##########################################################################
    #INITIAL CENTERS
    c1 = [10,10]
    c2 = [-10,-10]
    c = []
    c.append(c1)
    c.append(c2)

    #INITIALIZING COUNT
    count =0


    #CALLING THE KMEANS FUNCTION
    numpy1 = np.array([])
    numpy2 = np.array([])
    numpy3 = np.array([])
    int1 = int()
    numpy1, numpy2, numpy3, count = mykmeans(X,2,c,count)

    print("This is the final centers of the 2 clusters")
    print(numpy3)


    #PLOTTING GRAPH
    for x, y in numpy1:
        plt.plot(x, y, marker="o", color="blue", markeredgecolor="black")
    for x,y in numpy2 :
        plt.plot(x, y, marker="o", color="red", markeredgecolor="black")
        for x, y in numpy3:
            plt.plot(x, y, marker="x", color="black", markeredgecolor="black")
    plt.show()
    exit()

if __name__ == '__main__':
    main()


