# author: Hasan Masum(id: 1805052)
# task 2B: Symmetric Eigen Decomposition
import numpy as np

def symmetric_eigen(n: int, low:int = 1 , high: int = 100):
    # Produce a random n x n invertible symmetric matrix A. For the purpose of demonstrating, every cell of A will be an integer.
    # ref: https://stackoverflow.com/a/10806947
    A = np.random.randint(low, high, size=(n, n))
    A = A + A.T
    print("\n",f"Random {n} x {n} symmetric matrix A:")
    print(A, "\n")

    # Perform Eigen Decomposition using NumPy’s library function
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Print eigenvalues and eigenvectors
    print("\n", "Eigenvalues:")
    print(eigenvalues, "\n")
    print("\n", "Eigenvectors:")
    print(eigenvectors, "\n")

    # Reconstruct A from eigenvalues and eigenvectors (refer to Section 2.7). 
    # A = PDP^-1 because P is a matrix of eigenvectors and D is a diagonal matrix of eigenvalues
    # and D=P^-1AP
    A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T # for symmetric matrix, P^-1 = P^T
    print("\n", "Reconstructed matrix A:")
    print(A_reconstructed, "\n")

    # Check if the reconstruction worked properly. (np.allclose will come in handy.)
    print(f"Check if the reconstruction worked properly: {np.allclose(A, A_reconstructed)}")

if __name__ == "__main__":
    # take the dimensions of matrix n as input
    n = int(input("Enter the dimension of the matrix: "))
    symmetric_eigen(n, 1, 200)

