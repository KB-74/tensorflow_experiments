import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

COLUMNS = ['YearsExperience', 'Salary']
FEATURES = ['YearsExperience']
LABEL = 'Salary'

original_set = pd.read_csv('./resources/Salary_Data.csv',
                           skipinitialspace=True,
                           skiprows=1,
                           names=COLUMNS)

msk = np.random.rand(len(original_set)) < 0.8
training_set = original_set[msk]
test_set = original_set[~msk]

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

estimator = tf.estimator.LinearRegressor(
    feature_columns=feature_cols,
    model_dir="./models/sales")


def get_input_fn(data_set, num_epochs=None, n_batch=5, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle)


estimator.train(input_fn=get_input_fn(training_set),
                steps=100)

ev = estimator.evaluate(
    input_fn=get_input_fn(test_set))

prediction_set = pd.DataFrame(data={COLUMNS[0]: [1, 10], COLUMNS[1]: 0}, columns=COLUMNS)

predict_results = estimator.predict(
    input_fn=get_input_fn(prediction_set))

# Print the prediction results.
print("\nPrediction results:")
results_to_plot = {COLUMNS[0]: [], COLUMNS[1]: []}
for i, prediction in enumerate(predict_results):
    msg = "YearsExperience: {} Years, Prediction: ${: 9.2f}"
    xp = prediction_set[FEATURES[0]][i]
    salary = prediction["predictions"][0]
    msg = msg.format(xp, salary)
    print("    " + msg)
    results_to_plot[COLUMNS[0]] = xp
    results_to_plot[COLUMNS[1]] = salary

training_set.plot(x=COLUMNS[0], y=COLUMNS[1], kind='scatter')
plt.plot(results_to_plot[COLUMNS[0]], results_to_plot[COLUMNS[1]])
plt.show()
plt.close()
