import itertools

import pandas as pd
import tensorflow as tf
import numpy as np

COLUMNS = ['YearsExperience', 'Salary']
FEATURES = ['YearsExperience']
LABEL = 'Salary'

original_set = pd.read_csv('./resources/Salary_Data.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
msk = np.random.rand(len(original_set)) < 0.8
training_set = original_set[msk]
test_set = original_set[~msk]


print(training_set)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

estimator = tf.estimator.LinearRegressor(
    feature_columns=feature_cols,
    model_dir="./models/sales")


def get_input_fn(data_set, num_epochs=None, n_batch=128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle)


estimator.train(input_fn=get_input_fn(training_set, num_epochs=None, shuffle=False), steps=1000)

ev = estimator.evaluate(
    input_fn=get_input_fn(test_set,
                          num_epochs=1,
                          n_batch=128,
                          shuffle=False))

data = np.array([1, 0],[3, 0])

prediction_set = pd.DataFrame(data=data, columns=FEATURES)

print(training_set)
print(prediction_set)

y = estimator.predict(
    input_fn=get_input_fn(prediction_set,
                          num_epochs=1,
                          n_batch=128,
                          shuffle=False))

predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))

