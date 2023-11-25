# author: Hasn Masum(id: 1805052)
import cv2
import matplotlib.pyplot as plt
import numpy as np

# define min number of k
MIN_NUM_OF_K = 10
# Figure dimensions
FIG_WIDTH=12
FIG_HEIGHT=10

def reconstruct_image(image_path:str) -> np.ndarray:
    # Using OpenCV frameworks to read image.jpg.
    image = cv2.imread(image_path)

    print("image read done") 
    # Transforming image to grayscale using function cv2.cvtColor(). 
    A = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The grayscale image will be an n x m matrix A
    print("\n")
    print("Dimensions of the grayscale image:")
    print(A.shape, "\n")
    m = A.shape[0]
    n = A.shape[1]

    # Singular Value Decomposition using NumPy’s library function.
    # https://www.youtube.com/watch?v=CpD9XlTu3ys
    U, S, V_T = np.linalg.svd(A)

    # Print the dimensions of U, S, V
    print("Dimensions of U, S, V:")
    print(U.shape, S.shape, V_T.shape, "\n")


    # Given a matrix A and an integer k,
    # returns the k-rank approximation of A
    def low_rank_approximation(A: np.ndarray, k :int) -> np.ndarray:
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        V_T_k = V_T[:k, :]
        # A_k = U_k * S_k * V_T_k 
        return U_k @ S_k @ V_T_k # @ operator is shorthand for np.matmul() 

    # vary the value of k from 1 to min(n, m) (take at least 10 such values in the interval).
    upper_range = min(n, m) # upper range of k
    print("Value of k must be between 1 to", upper_range)
    print("If number of k is more than 10, then any input out of range will end the program.")

    axpproximations:np.ndarray = []
    k_values: int = []
    while True:
        k = int(input(f"Enter k between 1 to {upper_range} : "))
        if k < 1 or k > upper_range:
            if(len(k_values) > MIN_NUM_OF_K):
                print("Rendering image...")
                break
            print("Invalid input. Please try again.")
            continue
        k_values.append(k)
        # Compute the k-rank approximation of A
        A_k = low_rank_approximation(A, k)
        axpproximations.append(A_k)
        print(f"number of k {len(k_values)}")

    # Plot the k-rank approximation of the image
    # plt subplot 4 images per row
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    n_cols = 4
    n_rows = int(np.ceil(len(k_values) / n_cols)) # round up
    for n_k, A_k in enumerate(axpproximations):
        plt.subplot(n_rows, n_cols, n_k+1)
        plt.title(f"n_components = {k_values[n_k]}")
        plt.imshow(A_k, cmap="gray")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Take a photo of a book’s cover within your vicinity. Let’s assume it is named image.jpg
    reconstruct_image("image.jpg")