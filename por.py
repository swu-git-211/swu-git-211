from __future__ import annotations

from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator

def dummy_test():
    return 'branch_a'

with DAG(
    "por",
    default_args={
        "email": ["akira.sannam@g.swu.ac.th"],
    },
    description="A simple tutorial DAG",
    schedule_interval=None,  
    start_date=datetime(2021, 1, 1),
    tags=["example"],
) as dag:

    A_task = DummyOperator(task_id='branch_a', dag=dag)  
    B_task = DummyOperator(task_id='branch_false', dag=dag) 

    branch_task = BranchPythonOperator(
        task_id='branching',
        python_callable=dummy_test,
        dag=dag,
    )

    branch_task >> A_task
    branch_task >> B_task
