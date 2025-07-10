from airflow import models
from airflow.utils.dates import days_ago
from airflow.providers.google.cloud.operators.notebooks import NotebooksExecuteOperator

sklearn_nb_path = "gs://langchain_bucket_arseny/notebook/sklearn_notebook.ipynb"
transformers_nb_path = "gs://langchain_bucket_arseny/notebook/transformers_notebook.ipynb"

# DAG 
with models.DAG(
    "rag_embeddings_analysis_dag",
    schedule_interval=None,  
    start_date=days_ago(1),
    catchup=False,
    tags=["rag", "notebooks"],
) as dag:

    # sklearn notebook 
    run_sklearn = NotebooksExecuteOperator(
        task_id="run_sklearn_notebook",
        location="us-east1",
        project_id="grand-voltage-465301-e8",
        gcs_notebook_path=sklearn_nb_path,
    )

    # transformers notebook 
    run_transformers = NotebooksExecuteOperator(
        task_id="run_transformers_notebook",
        location="us-east1",
        project_id="grand-voltage-465301-e8",
        gcs_notebook_path=transformers_nb_path,
    )

    [run_sklearn, run_transformers]
