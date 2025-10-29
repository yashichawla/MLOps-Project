import pytest
from airflow.models import DAG, BaseOperator
from dags import salad_preprocess_dag


@pytest.fixture(scope="module")
def dag():
    """Fixture to return the DAG object once for all tests."""
    return salad_preprocess_dag.salad_preprocess_v1()


# ---------------------------------------------------------------------
# DAG-Level Structure Tests
# ---------------------------------------------------------------------

def test_dag_import_and_id(dag):
    """Ensure DAG loads and has the correct ID."""
    assert isinstance(dag, DAG)
    assert dag.dag_id == "salad_preprocess_v1"


def test_dag_has_description(dag):
    """DAG should have a non-empty description."""
    assert dag.description
    assert isinstance(dag.description, str)
    assert len(dag.description.strip()) > 0


def test_dag_default_args_exist(dag):
    """Ensure required default args are present."""
    defaults = dag.default_args
    for key in ["owner", "depends_on_past", "email_on_failure", "retries"]:
        assert key in defaults, f"Missing default arg: {key}"


def test_dag_retry_policy_valid(dag):
    """Retry settings should make sense."""
    args = dag.default_args
    assert args["retries"] >= 0
    if "retry_delay" in args:
        from datetime import timedelta
        assert isinstance(args["retry_delay"], timedelta)


def test_dag_schedule_and_start_date(dag):
    """Schedule and start_date should be defined."""
    assert dag.schedule_interval is not None
    assert dag.start_date is not None


# ---------------------------------------------------------------------
# Task-Level Tests
# ---------------------------------------------------------------------

def test_all_tasks_present(dag):
    """Ensure all expected tasks are present."""
    task_ids = [t.task_id for t in dag.tasks]
    expected = {
        "start",
        "preprocess_input_csv",
        "validate_output",
        "email_failure",
    }
    missing = expected - set(task_ids)
    assert not missing, f"Missing tasks: {missing}"


def test_task_types_are_operators(dag):
    """All tasks should be Airflow BaseOperator subclasses."""
    for task in dag.tasks:
        assert isinstance(task, BaseOperator), f"{task.task_id} is not a valid Operator"


def test_preprocess_to_validate_dependency(dag):
    """preprocess_input_csv should lead to validate_output."""
    preprocess = dag.get_task("preprocess_input_csv")
    downstream_ids = [t.task_id for t in preprocess.downstream_list]
    assert "validate_output" in downstream_ids


def test_validate_to_email_dependency(dag):
    """validate_output should lead to email_failure."""
    validate = dag.get_task("validate_output")
    downstream_ids = [t.task_id for t in validate.downstream_list]
    assert "email_failure" in downstream_ids


def test_start_task_has_downstream(dag):
    """Start task should trigger downstream tasks."""
    start = dag.get_task("start")
    assert len(start.downstream_list) > 0


# ---------------------------------------------------------------------
# DAG Graph & Serialization
# ---------------------------------------------------------------------

def test_dag_serializable(dag):
    """DAG should be serializable (Airflow parses it for web UI)."""
    json_repr = dag.to_json()
    assert isinstance(json_repr, str)
    assert dag.dag_id in json_repr


def test_dag_task_count_reasonable(dag):
    """Ensure DAG has an expected number of tasks."""
    assert len(dag.tasks) >= 3  # Start, preprocess, validate, email


def test_dag_no_circular_dependencies(dag):
    """DAG should have no circular dependencies."""
    for task in dag.tasks:
        for downstream in task.downstream_list:
            assert task.task_id not in [t.task_id for t in downstream.downstream_list], (
                f"Circular dependency detected between {task.task_id} and {downstream.task_id}"
            )


# ---------------------------------------------------------------------
# Failure Handling / Edge Behavior
# ---------------------------------------------------------------------

def test_email_failure_task_exists_and_has_no_downstream(dag):
    """email_failure should exist and not have downstream tasks."""
    email_task = dag.get_task("email_failure")
    assert email_task is not None
    assert len(email_task.downstream_list) == 0


def test_dag_owners_and_tags_defined(dag):
    """Owner and tags should be defined."""
    assert "owner" in dag.default_args
    assert hasattr(dag, "tags")
    assert isinstance(dag.tags, (list, tuple))


def test_dag_default_view_and_orientation(dag):
    """Check DAG UI visualization parameters."""
    assert hasattr(dag, "orientation")
    assert hasattr(dag, "default_view")
    assert dag.default_view in ["graph", "tree", "grid"]


def test_task_retry_and_email_settings(dag):
    """Each task should respect retry and email settings."""
    for task in dag.tasks:
        assert hasattr(task, "retries")
        assert task.retries >= 0
        if hasattr(task, "email_on_failure"):
            assert isinstance(task.email_on_failure, bool)


def test_dag_context_manager_construction():
    """DAG can be built as context manager without errors."""
    from airflow.utils.dates import days_ago
    from airflow.operators.dummy import DummyOperator

    with DAG(
        dag_id="context_dag_test",
        start_date=days_ago(1),
        schedule_interval="@daily",
        catchup=False,
    ) as dag:
        start = DummyOperator(task_id="start")
        end = DummyOperator(task_id="end")
        start >> end

    assert isinstance(dag, DAG)
    assert "start" in [t.task_id for t in dag.tasks]
