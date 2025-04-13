# Databricks notebook source
# https://docs.databricks.com/aws/en/mlflow/

# COMMAND ----------

# Notebook: 03_Model_Training
import mlflow
import mlflow.sklearn
from databricks import feature_store

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# COMMAND ----------

# --- Parameters ---
# These could be passed as widgets or job parameters
n_estimators = 150
max_depth = 10
random_state = 42

mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True, # Keep signature logging if desired
    log_models=False,  
    silent=True
)

# COMMAND ----------

# --- Load Data ---
# Assume 'prepared_df' (with target) and 'fs_table_name' are available or passed
# Option 1: Re-run previous steps if needed (not ideal for jobs)
# Option 2: Load base data and use Feature Store client to join features

dbutils.widgets.text("upstream_metrics_json", "{{tasks.DataCreation.values.notebook_output}}")
delta_lake_path = dbutils.widgets.get("upstream_metrics_json")

dbutils.widgets.text("upstream_metrics_json", "{{tasks.FeatureEngineering.values.notebook_output}}")
fs_table_name = dbutils.widgets.get("upstream_metrics_json")

if not delta_lake_path:
    delta_lake_path = '/mnt/adventureworks/prepared_data2'
    print(f"Delta Lake Path not found in history")

if not fs_table_name:
    fs_table_name = "databricks_us.adventureworks_db.sales_order_features2"
    print(f"Feature Store Table not found in history")

# Load base data again (or from Delta Lake) - need primary key and target
# This assumes 01_Data_Ingestion was run or data is accessible
try:
    base_df = spark.read.format("delta").load(delta_lake_path) # Example path
    # OR reload from DB if not saved
    # base_df = spark.read.jdbc(...) # As in notebook 01, select primary_key and target
    base_df = base_df.select("primary_key", "TotalDue") # Need target variable and key
except Exception as e:
    print(f"Could not load base data: {e}")
    dbutils.notebook.exit("Failed to load base data for training.")

# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().currentRunId()

# COMMAND ----------


fs = feature_store.FeatureStoreClient()

# Create Training Set by joining base data (target) with features from Feature Store
# Create a DataFrame with primary keys and timestamps (use OrderDate or current time)
# For simplicity, using just the primary key from our base_df
lookup_keys_df = base_df.select("primary_key")
lookup_keys_df.show(10)

# COMMAND ----------

display(base_df)

# COMMAND ----------

try:
    training_set = fs.create_training_set(
        df=base_df, # DataFrame containing labels and primary keys
        feature_lookups=[
            feature_store.FeatureLookup(
                table_name=fs_table_name,
                lookup_key="primary_key"
            )
        ],
        label="TotalDue",
        exclude_columns=["primary_key"] # Exclude non-feature/non-label columns
    )
    training_pd = training_set.load_df().toPandas() # Load data into Pandas for scikit-learn
    print("Training set created successfully.")

except Exception as e:
    print(f"Error creating training set from Feature Store: {e}")
    dbutils.notebook.exit("Feature Store training set creation failed.")

# COMMAND ----------

artifact_path="model",

# COMMAND ----------

# --- Train/Test Split ---
X = training_pd.drop("TotalDue", axis=1)
y = training_pd["TotalDue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# --- Model Training with MLflow ---
with mlflow.start_run() as run:
    # Log parameters (Autolog might capture some, but explicit is good too)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("feature_table", fs_table_name)

    # Train the model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Log metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Infer model signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, y_pred)

    # Log the model using the Feature Store API for better integration
    # This logs the model AND information about the features used
    fs.log_model(
        model=rf,
        artifact_path="model", # Name within MLflow run
        flavor=mlflow.sklearn,
        training_set=training_set, # Pass the training_set object
        signature=signature, # Include the inferred signature
        registered_model_name=None # Register in the next step
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print("Model logged with Feature Store context.")

# dbutils.notebook.exit(run.info.run_id)

# COMMAND ----------



# COMMAND ----------

# enable autologging
mlflow.sklearn.autolog()

# COMMAND ----------

# --- Model Training with MLflow ---
with mlflow.start_run() as run:
    # Train the model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    print("Model logged with Feature Store context.")
