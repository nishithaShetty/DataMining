from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from scipy.stats import norm

#creating GAUSSIANS DATASET
from pip._vendor.distlib.compat import raw_input

#EUCLIDEAN DISTANCES
def euclidean_distance(x1, x2):

    return np.sqrt(np.sum((x1 - x2) ** 2))

def sampleSet():
    mean1 = [1, 0]
    mean2 = [0,2.5]
    covar1 = [[0.9, 0.4], [0.4, 0.9]]
    covar2 = [[0.9, 0.4], [0.4, 0.9]]
    set1 = np.random.multivariate_normal(mean1, covar1, 500)
    set2 = np.random.multivariate_normal(mean2, covar1, 500)
    return set1,set2

def sampleSet2(mean,standardd):
    mu, sigma = mean, standardd  # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
    return s




def mykde(X,h):
    answer = np.size(X, axis=1)
    if(answer == 1):
        count2 = 0
        data_points2_pdf = np.array([])


        for row in X:
            i = row[0]
            data_points = np.array([])
            data_points1 = np.array([])
            count = 0
            half = h / 2
            lower = i - half
            upper = i + half
            for j in X:
                distance = abs(j - i)
                # if( lower<=j<= upper):
                if (distance < half):
                    data_points = np.append(data_points, j)
                    # a
                    # data_points1 = np.append(data_points1, distances)

                    count = count + 1

            count2 = count2 + 1
            p = count / 1000
            data_points2_pdf = np.append(data_points2_pdf, [p, i])
        data_points2_pdf = np.reshape(data_points2_pdf, (count2, 2))
        # print(data_points2_pdf)

        return (data_points2_pdf)
    else:
        count2 = 0
        data_points2_pdf = np.array([])

        for row in X:
            x1 = row[0]
            y1 = row[1]
            count = 0
            center = np.array([x1, y1])
            data_points = np.array([])
            data_points1 = np.array([])

            for row in X:
                x0 = row[0]
                y0 = row[1]
                points = np.array([x0, y0])
                distances = euclidean_distance(center, points)
                if (distances < h / 2):
                    data_points = np.append(data_points, [x0, y0])
                    count = count + 1
            data_points = np.append(data_points, [x1, y1])
            count = count + 1
            data_points = np.reshape(data_points, (count, 2))

            q = np.sum(a=data_points1, axis=0)

            p = (count) / 1000
            data_points2_pdf = np.append(data_points2_pdf, [x1, y1, p])

            count2 = count2 + 1

        data_points2_pdf = np.reshape(data_points2_pdf, (count2, 3))

        return (data_points2_pdf)


def main():

    k = raw_input("Enter Input 1: 1 Dimensional data, 2: 2Dimensional data : ")

    k = int(k)
    list =[0.1,1,5,10]

    if(k ==2):
      for h in list:
          b = sampleSet()
          X = np.concatenate([b[0], b[1]])
          a = mykde(X, h)
          for x, y, z in a:
              plt.plot(x, y, z, marker="o", color="blue", markeredgecolor="black")
          plt.show()

    elif(k==1):
        mean = raw_input("Enter mean ")
        mean = float(mean)
        sd = raw_input("Enter standard deviation ")
        sd = float(sd)

        for h in list:
            b = sampleSet2(mean, sd)
            b = np.reshape(b, (1000, 1))
            a = mykde(b, h)

            for x, y in a:
                plt.plot(y, x, marker="o", color="blue", markeredgecolor="black")
            plt.show()

    else:
        exit()


if __name__ == '__main__':
    main()

