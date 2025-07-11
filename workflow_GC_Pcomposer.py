from airflow import models
from airflow.utils.dates import days_ago
from airflow.providers.google.cloud.operators.notebooks import NotebooksExecuteOperator


BUCKET = "langchain_bucket_arseny"
SK_NB  = f"gs://{BUCKET}/notebook/sklearn_notebook.ipynb"
TF_NB  = f"gs://{BUCKET}/notebook/transformers_notebook.ipynb"

with models.DAG(
    dag_id="rag_embeddings_analysis_dag",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["rag", "notebooks"],
) as dag:

    run_sklearn = NotebooksExecuteOperator(
        task_id="run_sklearn_notebook",
        project_id="grand-voltage-465301-e8",
        location="us-central1",                 
        gcs_notebook_path=SK_NB,
    )

    run_transformers = NotebooksExecuteOperator(
        task_id="run_transformers_notebook",
        project_id="grand-voltage-465301-e8",
        location="us-central1",
        gcs_notebook_path=TF_NB,
    )

    [run_sklearn, run_transformers]
