from tensorflow.keras import layers, Sequential
from transformers import BertModel, BertForSequenceClassification
from preprocess import get_preproc_features, get_preproc_target
from tensorflow.keras.callbacks import EarlyStopping
from data import getting_yaml_data


data = getting_yaml_data(path_to_input_file='../raw_data/small conversational bot')
X_preproc = get_preproc_features(data['patterns'])
y_preproc = get_preproc_target(data['tags'])


def load_untrainable_model(num_labels=18):

    pretrained_model = BertForSequenceClassification(num_labels=num_labels)
    pretrained_model.trainable = False

    return pretrained_model

def create_model():

    pretrained_model = load_untrainable_model()


    model = Sequential()
    model.add(pretrained_model)
    model.add(layers.LSTM(20), activation="relu")
    model.add(layers.Dense(18, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model


def fit_model(model, X=X_preproc, y=y_preproc, epochs=30, batch_size=32, patience=20):


    es = EarlyStopping(patience=patience,restore_best_weights=True)

    fitted_model = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es],validation_split = 0.1)

    return fitted_model


if __name__== '__main__':
    model = create_model()
    print(model.summary)
