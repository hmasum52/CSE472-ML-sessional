from train_1805052 import *

def load_test_data_set():
    # return ds.EMNIST(
    #     root="./data",
    #     split="letters",
    #     train=False,
    #     transform=transforms.ToTensor(),
    # )
    with open('ids7.pickle', 'rb') as ids7:
        independent_test_dataset = pickle.load(ids7)
    return independent_test_dataset


def test(model_path="model_1805052.pkl"):
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


def test_best(model_path="model_1805052-best.pkl"):
    # test_ds = load_test_data_set()
    test_ds = load_test_data_set()

    print("data loaded")

    test_ds = preprocess_data_set(test_ds)

    print("data preprocessed")

    X_test = test_ds.data
    y_test = test_ds.targets

    lrate=0.005
    input_size = 28 * 28
    output_size = 26

    model = FNN(
        loss=CategoricalCrossEntropyLoss(),
        optimizer=SGD(learning_rate=lrate),
        learning_rate=lrate,
        layers=[
            Flatten(),
            DenseLayer(input_size, 1024, ReLU()),
            DropoutLayer(dropout_rate=0.3),
            DenseLayer(1024, 512, ReLU()),
            DropoutLayer(dropout_rate=0.2),
            DenseLayer(512, 64, ReLU()),
            DropoutLayer(dropout_rate=0.1),
            DenseLayer(64, output_size, Softmax()),
        ]
    )

    model.load_best_model(model_path)

    print("model loaded")

    test_loss, test_accuracy, f1 = model.best_model_evaluate(X_test, y_test)


    print(f"test loss: {test_loss}, test accuracy: {test_accuracy}")

    print(f"f1 score: {f1}")


if __name__ == '__main__':
    #test()
    test_best()