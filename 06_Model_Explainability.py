# Databricks notebook source
# https://docs.databricks.com/aws/en/mlflow/

# COMMAND ----------

# MAGIC %pip install shap lime

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
import matplotlib.pyplot as plt # Needed for saving SHAP plots

# Import SHAP and LIME libraries
import shap
import lime
import lime.lime_tabular


# COMMAND ----------

# --- Parameters ---
# These could be passed as widgets or job parameters
n_estimators = 150
max_depth = 10
random_state = 42

mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True, # Keep signature logging if desired
    log_models=False,  # IMPORTANT: Disable autologging models for fs.log_model
    silent=True
)

# COMMAND ----------

# --- Load Data ---
# Assume 'prepared_df' (with target) and 'fs_table_name' are available or passed
# Option 1: Re-run previous steps if needed (not ideal for jobs)
# Option 2: Load base data and use Feature Store client to join features

# Load base data again (or from Delta Lake) - need primary key and target
# This assumes 01_Data_Ingestion was run or data is accessible
try:
    # Using the path from your example code
    base_df = spark.read.format("delta").load("/mnt/adventureworks/prepared_data2")
    base_df = base_df.select("primary_key", "TotalDue") # Need target variable and key
    print(f"Loaded base data for training. Count: {base_df.count()}")
except Exception as e:
    print(f"Could not load base data: {e}")
    dbutils.notebook.exit("Failed to load base data for training.")

# COMMAND ----------

# Using the feature table name from your example code
fs_table_name = "databricks_us.adventureworks_db.sales_order_features2"

fs = feature_store.FeatureStoreClient()

# Create Training Set by joining base data (target) with features from Feature Store
# Create a DataFrame with primary keys and timestamps (use OrderDate or current time)
# For simplicity, using just the primary key from our base_df
lookup_keys_df = base_df.select("primary_key")
print("Lookup keys head:")
lookup_keys_df.show(5) # Check keys

# COMMAND ----------


print("Base data head:")
display(base_df.limit(5)) # Check base data

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
    print(f"Training data shape: {training_pd.shape}")

except Exception as e:
    print(f"Error creating training set from Feature Store: {e}")
    dbutils.notebook.exit("Feature Store training set creation failed.")


# COMMAND ----------

artifact_path="model",

# COMMAND ----------

# --- Train/Test Split ---
# Ensure 'TotalDue' exists before dropping
if 'TotalDue' not in training_pd.columns:
     raise ValueError("Target variable 'TotalDue' not found in the training set DataFrame.")

X = training_pd.drop("TotalDue", axis=1)
y = training_pd["TotalDue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# --- Model Training, Logging, and Explainability ---
with mlflow.start_run() as run:
    # Log parameters (Autolog might capture some, but explicit is good too)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("feature_table", fs_table_name)

    # Train the model
    print("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)
    print("Training complete.")

    # Make predictions
    y_pred_train = rf.predict(X_train) # For signature inference if needed later
    y_pred_test = rf.predict(X_test)

    # Log metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    print(f"Metrics logged: RMSE={rmse}, R2={r2}")

    # Infer model signature using test set prediction for better representation
    from mlflow.models.signature import infer_signature
    # Use X_train and corresponding predictions for signature
    signature = infer_signature(X_train, y_pred_train)
    print("Model signature inferred.")

    # Log the model using the Feature Store API for better integration
    print("Logging model with Feature Store context...")
    fs.log_model(
        model=rf,
        artifact_path="model", # Name within MLflow run artifacts
        flavor=mlflow.sklearn,
        training_set=training_set, # Pass the training_set object
        signature=signature, # Include the inferred signature
        registered_model_name=None # Register in the next step
    )
    print("Model logged successfully using fs.log_model.")

# COMMAND ----------

# ========================================
# ===== SHAP Analysis ===================
# ========================================
print("Starting SHAP analysis...")
# Use TreeExplainer for efficiency with RandomForest
explainer = shap.TreeExplainer(rf)
# Calculate SHAP values for the test set (can use a subset for large data)
shap_values = explainer.shap_values(X_test)
print("SHAP values calculated.")

# --- Log SHAP Summary Plot ---
print("Generating and logging SHAP summary plot...")
shap_summary_plot_path = "/tmp/shap_summary_plot.png"
# Create the plot - use show=False to prevent direct display here
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig(shap_summary_plot_path, bbox_inches='tight') # Save the plot
plt.close() # Close plot to prevent double display
mlflow.log_artifact(shap_summary_plot_path, "shap_plots") # Log as artifact
print("SHAP summary plot logged.")

# --- Log SHAP Dependence Plots (Example for top 2 features) ---
# Get global feature importance from mean absolute SHAP values
vals = np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X_train.columns, vals)), columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
top_features = feature_importance['col_name'].head(2).tolist() # Get top 2 feature names

for feature in top_features:
    print(f"Generating and logging SHAP dependence plot for: {feature}")
    shap_dependence_plot_path = f"/tmp/shap_dependence_plot_{feature}.png"
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.savefig(shap_dependence_plot_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(shap_dependence_plot_path, "shap_plots")
print("SHAP dependence plots logged.")

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
