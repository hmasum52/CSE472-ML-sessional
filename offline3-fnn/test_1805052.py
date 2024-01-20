from train_1805052 import *

def load_test_data_set():
    return ds.EMNIST(
        root="./data",
        split="letters",
        train=False,
        transform=transforms.ToTensor(),
    )

def test(model_path="last.pkl"):
    test_ds = load_test_data_set()

    print("data loaded")

    test_ds = preprocess_data_set(test_ds)

    print("data preprocessed")

    X_test = test_ds.data
    y_test = test_ds.targets

    model = FNN.load(model_path)

    print("model loaded")

    test_loss, test_accuracy, f1 = model.evaluate(X_test, y_test)

    print(f"test loss: {test_loss}, test accuracy: {test_accuracy}")

    print(f"f1 score: {f1}")

if __name__ == '__main__':
    test()