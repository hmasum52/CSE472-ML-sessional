# read file and make N*M numpy matrix
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

FIGURE_PATH="figure"

def read_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split(",") for line in lines]
    lines = [[float(x) for x in line] for line in lines]
    return lines

def get_data(file_name):
    lines = read_file(file_name)
    data = np.array(lines)
    return data

# plot the data points along two principal axes(if pca is applied) or along two feature dimensions(if PCA is not applied)
import matplotlib.pyplot as plt
import os
def plot_data(data, title, save=False):
    global FIGURE_PATH
    plt.scatter(data[:, 0], data[:, 1], color="black", s=10)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")

    if save:
        # make sure to create a folder named "figure" in the current directory
        if not os.path.exists(FIGURE_PATH):
            os.makedirs(FIGURE_PATH)
        plt.savefig(f"{FIGURE_PATH}/{title}.png")
        plt.close()
    plt.show()

##############################################
    # PCA using covariance matrix
##############################################

# pca using np svd
# sklearn uses svd to calculate 
# https://gregorygundersen.com/blog/2018/12/10/svd/
# https://www.educative.io/blog/sign-ambiguity-in-singular-value-decomposition
def svd_flip(u, v):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v

def pca_svd(data, n_dims=6):
    # subtract mean i.e mean centering
    centered_data = data - np.mean(data, axis=0)
    # calculate covariance matrix with svd
    U , S, eig_vec_T = np.linalg.svd(centered_data, full_matrices=False) #  returns U, S , V.T (V.T is eigen vectors)

    # singular values from the SVD are the square roots of the eigenvalues from the covariance matrix of the data
    eig_val = S ** 2 / centered_data.shape[0]
    
    # flip eigen vector's sign to enforce deterministic output
    U, eig_vec_T = svd_flip(U, eig_vec_T)
    
    # project data on eigen vectors
    transformed_data = np.dot(centered_data, eig_vec_T.T)
    # select first n_dims columns
    return transformed_data[:, :n_dims], eig_val[:n_dims], eig_vec_T[:, :n_dims]


#########################
# Gaussian Mixture Model
#########################

# import scipy multivariate normal distribution
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
  def __init__(self, n_components: int):
    self.n_components: int = n_components
    self.n_features: int = 2
    self.means: np.ndarray = None # mu shape = (n_components, n_features)
    self.covariances: np.ndarray = None  # sigma shape = (n_components, n_features, n_features)
    self.weights: np.ndarray = None
    self.log_likelihood: np.ndarray = None
    print(f"GaussianMixtureModel for {self.n_components} components...")

  def _init_params(self, X: np.ndarray) -> None:

    # initialize number of features
    self.n_features = X.shape[1]

    # initialize means
    self.means = np.random.rand(self.n_components, self.n_features)

    # initialize covariances
    # shape = (n_components, n_features, n_features)
    self.covariances = np.array([np.eye(self.n_features)] * self.n_components)

    # initialize weights
    self.weights = np.ones(self.n_components) / self.n_components

    # initialize log likelihood
    self.log_likelihood = []


  def fit(self, X, n_itr = 1000, tol=1e-3, plot=False, contour=False, countour_itr=1, save_final=False):
    global FIGURE_PATH
    # initialize parameters
    self._init_params(X)

    if plot: plt.ion()

    for itr in range(n_itr):
      # E step
      # calculate probabilities of each data point belonging to each component
      component_prob_mat = self._e_step(X)
      # M step
      # update parameters
      self._m_step(X, component_prob_mat)

      # calculate log likelihood
      log_likelihood = self._log_likelihood(X)
      self.log_likelihood.append(log_likelihood)

      # check for convergence
      if len(self.log_likelihood) > 2 and np.abs(self.log_likelihood[-1] - self.log_likelihood[-2]) < tol:
        break

      # plot
      if plot and self.n_features >= 2 and itr % countour_itr == 0:
        plt.clf()
        plt.title(f"Iteration {itr}")
        plt.scatter(X[:, 0], X[:, 1], c=component_prob_mat.argmax(axis=1), s=10, cmap='viridis')
        if contour:
          self._plot_contour(X)
        plt.pause(0.01)

    if plot: 
      plt.ioff()
      # plot final result
      plt.clf()
      plt.title(f"Iteration {itr}")
      plt.scatter(X[:, 0], X[:, 1], c=component_prob_mat.argmax(axis=1), s=10, cmap='viridis')
      # plot mean with red cross
      plt.scatter(self.means[:, 0], self.means[:, 1], marker='x', c='red', s=30)
      if contour:
        self._plot_contour(X)
      # save figure
      if save_final:
        if not os.path.exists(FIGURE_PATH):
          os.makedirs(FIGURE_PATH)
        plt.savefig(f"{FIGURE_PATH}/GMM-{self.n_components}-components-fit.png")
      plt.show()

    print(f"Converged after {itr:3d}/{n_itr} iterations, likelihood={self.log_likelihood[-1]:.4f}")

  def plot_cluster(self, X, mean=False, save=False):
    global FIGURE_PATH
    plt.title(f"GMM with {self.n_components} components")
    plt.scatter(X[:, 0], X[:, 1], c=self.predict(X), s=10, cmap='viridis')
    # plot mean with red cross
    if mean: 
      plt.scatter(self.means[:, 0], self.means[:, 1], marker='x', c='red', s=30)
    if save:
      # make sure to create a folder named "figure" in the current directory
      if not os.path.exists(FIGURE_PATH):
          os.makedirs(FIGURE_PATH)
      plt.savefig(f"{FIGURE_PATH}/GMM-{self.n_components}-components.png")
      plt.close()
    plt.show()

  def _e_step(self, X: np.ndarray) -> np.ndarray:
    # describe the probability that a given data point belongs to a particular Gaussian component in the mixture model.
    # a measure of how much a particular component is "responsible" for a given data point.
    component_prob_mat = np.zeros((X.shape[0], self.n_components), dtype=float) # shape = (n_samples, n_components)
    
    for i in range(self.n_components):
      normal_dist = multivariate_normal.pdf(
        X, 
        mean=self.means[i], 
        cov=self.covariances[i], 
        allow_singular=True) + 1e-6 # shape = (n_samples, )
      # p_ij = P(C = i) * P(x_j | C = i, mu_i, sigma_i)
      component_prob_mat[:, i] = self.weights[i] * normal_dist

    # calculate mixture likelihood
  
    # P(x) = sum_i_1_to_k[ P(C = i) * P(x | C = i) ]
    #      = sum_i_1_to_k[ w_i * N(x | mu_i, sigma_i) ]
    self.mixture_likelihood:np.ndarray = component_prob_mat.sum(axis=1, keepdims=True) # shape = (n_samples, 1)

    # normalize responsibility to make sure that each row of responsibility matrix sums to 1
    # each cell represents the probability that a given data point belongs 
    # to a particular Gaussian component in the mixture model.
    component_prob_mat /= component_prob_mat.sum(axis=1, keepdims=True)

    return component_prob_mat
  
  def _m_step(self, X: np.ndarray, component_prob_mat: np.ndarray) -> None:
    
    for component_i in range(self.n_components):
      # get probability of each data point belonging to ith component
      component_i_probs:np.ndarray = component_prob_mat[:, component_i].reshape(-1, 1) # shape = (n_samples, 1). 
      # sum of all probabilities of data points belonging to ith component
      # n_i
      total_prob = component_i_probs.sum()
  
      # update mean
      # mu_i = sum_j(p_ji * x_j) / n_i
      self.means[component_i] = np.sum(component_i_probs * X, axis=0) / total_prob

      # update covariance
      deviation:np.ndarray = X - self.means[component_i] # shape = (n_samples, n_features)
      self.covariances[component_i] = np.dot(deviation.T, component_i_probs * deviation) / total_prob

      # update weights
      # w_i = n_i / N
      self.weights[component_i] = total_prob / X.shape[0]
  
  def _log_likelihood(self, X):
    return np.sum(np.log(self.mixture_likelihood))
  
  def _plot_contour(self, X):
    # plot contour
    n = X.shape[0]
    x = np.linspace(X[:, 0].min(), X[:, 0].max()+1,n)
    y = np.linspace(X[:, 1].min(), X[:, 1].max()+1,n)
    X, Y = np.meshgrid(x, y)
    XX = np.dstack([X, Y])
    pdfs = []
    for i in range(self.n_components):
      pdfs.append( 
        self.weights[i]*multivariate_normal.pdf(XX, mean=self.means[i], cov=self.covariances[i], allow_singular=True))
    
    for z in pdfs:
      plt.contour(X, Y, z, levels=5)
      
    

  def predict(self, X):
    # calculate probabilities of each data point belonging to each component
    component_prob_mat = self._e_step(X)
    # return index of component with max probability
    return np.argmax(component_prob_mat, axis=1)
  

def plot_likelihood(log_likelihood,  save=False):
  global FIGURE_PATH
  plt.plot(range(3, 9),  log_likelihood)
  plt.title("Log Likelihood")
  plt.xlabel("K")
  plt.ylabel("Log Likelihood")
  if save:
      # make sure to create a folder named "figure" in the current directory
      if not os.path.exists(FIGURE_PATH):
          os.makedirs(FIGURE_PATH)
      plt.savefig(f"{FIGURE_PATH}/Log Likelihood.png")
      plt.close()

  plt.show()

import sys

def main():
    global FIGURE_PATH

    # take file name from argument
    input_file = "dataset/100D_data_points.txt"
    if len(sys.argv) > 1:
      input_file = sys.argv[1]  

    print(f"Reading data from {input_file}...\n") 

    FIGURE_PATH = f"figure/{input_file.split('.')[0].split('/')[1]}"
    
    # read data
    data = get_data(input_file)
    # plot data
    # plot_data(data, "Original Data")
    # apply pca
    if data.shape[1] > 2:
      X, eig_val, eig_vec = pca_svd(data.copy(), n_dims=2)
    else:
      X = data.copy()
    
    # plot pca data
    plot_data(X, "PCA Data", save=True)
    # apply gmm
    best_likelihood_list = [] 
    for k in range(3, 9):
      best_gmm = None
      best_log_likelihood = -np.inf
      for i in range(5):
        gmm = GaussianMixtureModel(n_components=k)
        gmm.fit(X, plot=False)
        if gmm.log_likelihood[-1] > best_log_likelihood:
          best_log_likelihood = gmm.log_likelihood[-1]
          best_gmm = gmm
      best_likelihood_list.append(best_log_likelihood)
      print("best likelihood:", best_log_likelihood, "for k =", best_gmm.n_components)
      best_gmm.plot_cluster(X, mean=False, save=True)

    # best_gmm = GaussianMixtureModel(n_components=3)
    # best_gmm.fit(X, plot=True, contour=True, countour_itr=1)
    # best_log_likelihood = best_gmm.log_likelihood[-1]

    # plot log likelihood
    plot_likelihood(best_likelihood_list, save=True)
    # plot cluster
    

if __name__ == "__main__":
    main()