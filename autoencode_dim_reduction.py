from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pickle import dump
import argparse

def get_data(training_data):
    """
    read training.csv and return the X,y as series
    :return: X - the data representing the road view
             y - what turn value
    """
    df = pd.read_csv(training_data, header=None)
    print(f"Training Data Shape: {df.shape}")
    # print(df.head())
    X = df.loc[:, 1:]
    y = df.loc[:, 0]
    print(X.shape)
    # print(y.shape)
    return X, y


def main(min_dims, training_data):
    X, y = get_data(training_data)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    input_size = X.shape[1]
    layer_1 = 128
    layer_2 = 64
    layer_3 = min_dims

    # Create Encoder
    encoder = Sequential()
    encoder.add(Dense(layer_1, input_shape=(input_size,), activation="relu"))
    encoder.add(Dense(layer_2,activation="relu"))
    encoder.add(Dense(layer_3,activation="relu"))


    # Create Decoder
    decoder = Sequential()
    decoder.add(Dense(layer_2, input_shape=(layer_3,), activation="relu"))
    decoder.add(Dense(layer_1, activation="relu"))
    decoder.add(Dense(input_size, activation="relu"))

    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(loss="mse", optimizer=SGD(learning_rate=0.1))

    autoencoder.fit(X_scaled, X_scaled, epochs=500)

    losses = autoencoder.history.history
    pd.DataFrame(losses).plot()
    plt.show()

    # use encoder to compress or remove dimensionality
    x_dim_reduced = encoder.predict(X_scaled)

    print(f"Dim Reduced Shape: {x_dim_reduced.shape}")

    df_dim_reduced = pd.concat([y, pd.DataFrame(x_dim_reduced)], axis=1)
    print(df_dim_reduced.shape)
    df_dim_reduced.to_csv(f"training_{min_dims}.csv", index=None, header=None)

    # save the model
    encoder.save("encoder_model.h5")
    print(encoder.summary())
    # save the scaler
    dump(scaler, open('encoder_scaler.pkl', 'wb'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dims", required=False, type=int, default=32, help="Minimum Number of Dimensions")
    ap.add_argument("--training-data", required=False, default='training.csv', help="FQP to training data csv file")

    args = vars(ap.parse_args())

    mdims = args['min_dims']
    train_data = args['training_data']

    print(f"Minimum Encoder Dimension: {mdims}")
    print(f"Training Data File: training_{mdims}.csv")

    main(mdims, train_data)



