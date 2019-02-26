from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["jobName_1", "jobName_2", "userName", "queue", "numProcessors"]
CONTINUOUS_COLUMNS = ["memReq", "cpuTime"]

def build_estimator(model_type,
                    model_optimizer,
                    model_dir,
                    learning_rate,
                    l1_regular,
                    l2_regular):
    """Build an estimator."""

    # Sparse base columns.
    queue = tf.contrib.layers.sparse_column_with_keys(column_name="queue",
                                                      keys=["qosH", "qosC"])
    num_processors = tf.contrib.layers.sparse_column_with_keys(
        column_name="numProcessors",
        keys=["1", "2",  "3", "4", "8", "16"])
    job_name_1 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "jobName_1", hash_bucket_size=1000)
    job_name_2 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "jobName_2", hash_bucket_size=1000)
    user_name = tf.contrib.layers.sparse_column_with_hash_bucket(
        "userName", hash_bucket_size=1000)

    # Continuous base columns.
    mem_req = tf.contrib.layers.real_valued_column("memReq")
    cpu_time = tf.contrib.layers.real_valued_column("cpuTime")

    wide_columns = [queue, job_name_1, job_name_2, user_name,
                    mem_req, num_processors, cpu_time]
    deep_columns = [
        tf.contrib.layers.embedding_column(queue, dimension=8),
        tf.contrib.layers.embedding_column(job_name_1, dimension=8),
        tf.contrib.layers.embedding_column(job_name_2, dimension=8),
        tf.contrib.layers.embedding_column(user_name, dimension=8),
        tf.contrib.layers.embedding_column(num_processors, dimension=8),
        mem_req,
        cpu_time,
    ]
    wide_columns_w_n_d = [queue, job_name_1, job_name_2, user_name, num_processors]

    if model_optimizer == "ProximalAdagrad":
        model_optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate=learning_rate,
            l1_regularization_strength=l1_regular,
            l2_regularization_strength=l2_regular)
    elif model_optimizer == "Adam":
        model_optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
    else:
        model_optimizer = tf.train.FtrlOptimizer(
            learning_rate=learning_rate
        )

    if model_type == "wide":
        m = tf.contrib.learn.LinearRegressor(model_dir=model_dir,
                                             feature_columns=wide_columns,
                                             optimizer=model_optimizer)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNRegressor(model_dir=model_dir,
                                          feature_columns=deep_columns,
                                          hidden_units=[10, 10],
                                          optimizer=model_optimizer)
    else:
        m = tf.contrib.learn.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            linear_feature_columns=wide_columns_w_n_d,
            linear_optimizer=model_optimizer,
            dnn_feature_columns=deep_columns,
            dnn_optimizer=model_optimizer,
            dnn_hidden_units=[300,300],
            fix_global_step_increment_bug=True)
    return m

def input_fn(df):
    """Input builder function."""

    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

def rescale(s):
    reshaped_s = s.values.reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(reshaped_s)
    return pd.DataFrame(data=scaler.transform(reshaped_s)), scaler

input_data="./donejobs.cleaned.csv"
df = pd.read_csv(
    tf.gfile.Open(input_data),
    skipinitialspace=True,
    engine="python")
# Split the 'jobName' column into two columns: 'jobName_1', 'jobName_2'
df_added_features = \
    df['jobName'].str.split(
        ":",
        expand=True)[[1, 2]].rename(columns={1: "jobName_1",
                                             2: "jobName_2"})

model_type="wide_n_deep"
model_optimizer="Adam"
model_dir="./model"
train_steps=6000
learning_rate=0.01
l1_regular=0.01
l2_regular=0.001

df['numProcessors'] = df['numProcessors'].apply(str)
# Combine the original columns with the two additional columns
# Standardize numerical columns

#df['memReq'], mem_req_scaler = rescale(df['memReq'])
#df['cpuTime'], cpu_time_scaler = rescale(df['cpuTime'])
#df['maxMem'], max_mem_scaler = rescale(df['maxMem'])
# Specify the label column
df[LABEL_COLUMN] = df["maxMem"]
df = pd.concat([df, df_added_features], axis=1)
# Drop rows with NA values
df = df.dropna(how='any', axis=0)
# Split the train, test, and prediction data
df_train = df[:int(df.shape[0] * 0.8)]
df_test = df[int(df.shape[0] * 0.8):]
df_predict = df_test.copy()
# Build the model
m = build_estimator(model_type,
                    model_optimizer,
                    model_dir,
                    learning_rate,
                    l1_regular,
                    l2_regular)
# Optimize
m.fit(input_fn=lambda: input_fn(df_test),
      steps=train_steps)
# Evaluate
results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
# Predict
predictions = list(m.predict(input_fn=lambda: input_fn(df_test)))
# Convert normalized data back to its original scale
df_predict['predict_maxMem'] = predictions

plt.figure(figsize=(12,8))
r = sns.tsplot(df_predict['maxMem'][100:1100], color="red")
b = sns.tsplot(df_predict['predict_maxMem'][100:1100], color="blue")
g = sns.tsplot(abs(df_predict['maxMem'] - df_predict['predict_maxMem'])[100:1100], color="green")
plt.savefig("./dnn_result.png")
plt.show()

plt.figure(figsize=(12,8))
r = sns.tsplot(df_predict['maxMem'][200:400], color="red")
b = sns.tsplot(df_predict['predict_maxMem'][200:400], color="blue")
g = sns.tsplot(abs(df_predict['maxMem'] - df_predict['predict_maxMem'])[200:400], color="green")
plt.savefig("./dnn_result_detail.png")
plt.show()