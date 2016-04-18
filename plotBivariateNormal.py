__author__ = 'seanhendryx'
# Sean Hendryx
# Script built to run on Python version 2.7.11
# References:
# Scipy documentation http://docs.scipy.org/doc/scipy-0.14.0/reference/index.html
# Wikipedia: Multivariate Normal Distribution https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
# Wikipedia Covariance Matrix https://en.wikipedia.org/wiki/
# StatsModels Statistics in Python http://statsmodels.sourceforge.net/0.6.0/_modules/statsmodels/stats/moment_helpers.html
# Bivariate Normal Distribution Wolfram Math World http://mathworld.wolfram.com/BivariateNormalDistribution.html
# http://lmf-ramblings.blogspot.com/2009/07/multivariate-normal-distribution-in.html
# http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
# http://www.extremeoptimization.com/Documentation/Statistics/Multivariate_Distributions/Multivariate_Normal_Distribution.aspx
# Probabilistic Graphical Models by Koller and Friedman


import numpy
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D

def main():
    """
    Outputs 3D graphic of multivariate normal distribution.
    :return: none
    """
    mu = [0,0]
    #A covariance matrix has been used to enable implementation in more dimensions than the bivariate case.  Sigma is calculated below and printed for reference.
    covarianceMatrix = [[.5,.3], [.3,2.0]]
    sigma = numpy.sqrt(numpy.diag(covarianceMatrix))
    print "sigma: ", sigma
    plot_mv_normal(mu, covarianceMatrix)

def mv_normal(x, mu, covarianceMatrix):
    """
    :param x: Input values (arrays) to use in the calculation of the multivariate Gaussian probability density
    :param mu: mean
    :param covarianceMatrix: input covariance matrix that determines how the variables covary
    :return: The probability densities from the multivariate normal distribution
    """
    #k = the dimension of the space where x takes values
    k = x.shape[0]
    #distance is the "distance between x and mean:
    distance = x-mu
    #Covariance matrix as specified in assignment prompt:

    firstFragment = numpy.exp(-0.5*k*numpy.log(2*numpy.pi))
    secondFragment = numpy.power(numpy.linalg.det(covarianceMatrix),-0.5)
    thirdFragment = numpy.exp(-0.5*numpy.dot(numpy.dot(distance.transpose(),numpy.linalg.inv(covarianceMatrix)),distance))

    multivariateNormalDistribution = firstFragment*secondFragment*thirdFragment
    return multivariateNormalDistribution

def plot_mv_normal(mu, covarianceMatrix, xmin=-5, xmax=5, ymin=-5, ymax=5, resolution=50):
    X, Y = numpy.meshgrid(numpy.linspace(xmin, xmax, resolution),
                          numpy.linspace(ymin, ymax, resolution))

    # 2d array to store mv_normal pdf values
    pdfv = numpy.zeros(X.shape)

    # calculate pdf values at each X,Y location
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdfv[i][j] = mv_normal(numpy.array([X[i][j], Y[i][j]]), mu, covarianceMatrix)

    # Contour plot
    plt.figure()
    plt.contour(X, Y, pdfv)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, pdfv, rstride=1, cstride=1,
                           cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    fig.colorbar(surf)

    plt.show()

# Main Function
if __name__ == '__main__':
    main()