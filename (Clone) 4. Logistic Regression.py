# Databricks notebook source
# MAGIC %run "./0. Set Config"

# COMMAND ----------

mlflow_run_name = "{}_logistic_run".format(config['model name'])

# COMMAND ----------

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from mlflow.models import infer_signature
from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

# 피쳐 스토어 데이터 로드
fs = FeatureStoreClient()
df = fs.read_table(name=user_feature_table)
df = df.toPandas()
df.head()

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# COMMAND ----------

X = df.drop(columns=['Survived', 'PassengerId', 'Sex', 'Embarked'])
y = df['Survived']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2024, shuffle=True)

# COMMAND ----------

with mlflow.start_run(run_name=mlflow_run_name) as run:
    random_state = 2024
    dtc = LogisticRegression(random_state=random_state)
    dtc.fit(train_x, train_y)
    y_pred_class = dtc.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred_class)
    f1 = f1_score(test_y, y_pred_class)

    mlflow.log_param('random_state', random_state)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_score', f1)
    
    signature = infer_signature(train_x, train_y)
    example = test_x[0:1]
    mlflow.sklearn.log_model(dtc, 'model', signature=signature, input_example=example)    


# COMMAND ----------


