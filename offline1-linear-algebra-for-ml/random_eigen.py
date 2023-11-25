# author: Hasan Masum(id: 1805052)
# task 2A
import numpy as np

def random_eigen(n: int, low:int = 1 , high: int = 100):
    # Produce a random n x n invertible matrix A. For the purpose of demonstrating, every cell of A will be an integer.
    while True:
        A = np.random.randint(low, high, size=(n, n))
        if np.linalg.det(A) != 0:
            break

    # Print the matrix A
    print("Matrix A:")
    print(A, "\n")

    # Perform Eigen Decomposition using NumPy’s library function 
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Reconstruct A from eigenvalues and eigenvectors 
    # A = PDP^-1 because P is a matrix of eigenvectors and D is a diagonal matrix of eigenvalues
    # and D=P^-1AP
    A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)

    print("Reconstructed Matrix A:")
    # convert A_reconstructed to int and real
    A_r = np.array([np.abs(A_reconstructed[i]) for i in range(len(A_reconstructed))])
    print(A_r, "\n")

    # Check if the reconstruction worked properly. 
    # (np.allclose(): Returns True if two arrays are element-wise equal within a tolerance.)
    print(f"Is A == A_reconstructed ? {np.allclose(A, A_reconstructed)}")
    

# ref: http://www.ltcconline.net/greenl/courses/203/MatrixOnVectors/diagonalization.htm?#:~:text=The%20j%20th%20column%20of%20AP%20equals%20the,nonsingular%20so%20that%20D%20%3D%20P%20-1%20AP
if __name__ == "__main__":
    # take the dimensions of matrix n as input
    n = int(input("Enter the dimension of the matrix: "))
    random_eigen(n, 1, 200)