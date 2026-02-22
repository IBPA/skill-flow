"""Tests for analyze_failure_modes module."""

import json

import pytest
from analysis.failure.core import (
    TaskData,
    compress_trajectory,
    extract_amenability,
    extract_one_liner,
    format_test_results,
    list_task_dirs,
    load_task_data,
)
from analysis.failure.main import (
    generate_aggregate_report,
    write_task_summary,
)


@pytest.fixture
def sample_trajectory_steps():
    """Sample trajectory steps for testing."""
    return [
        {"step_id": 1, "source": "user", "message": "Test task prompt"},
        {
            "step_id": 2,
            "source": "agent",
            "message": "Starting analysis",
            "tool_calls": [
                {
                    "function_name": "shell",
                    "arguments": {"command": ["bash", "-lc", "ls -la"]},
                }
            ],
            "observation": {"results": [{"content": "file1.txt\nfile2.txt"}]},
        },
        {
            "step_id": 3,
            "source": "agent",
            "message": "Creating solution",
            "tool_calls": [{"function_name": "write_file", "arguments": {}}],
        },
    ]


@pytest.fixture
def sample_test_results():
    """Sample CTRF test results for testing."""
    return {
        "results": {
            "summary": {"tests": 2, "passed": 1, "failed": 1},
            "tests": [
                {"name": "test_one", "status": "passed"},
                {
                    "name": "test_two",
                    "status": "failed",
                    "trace": "AssertionError: expected 5, got 3",
                },
            ],
        }
    }


@pytest.fixture
def sample_task_data(sample_trajectory_steps, sample_test_results):
    """Sample TaskData for testing."""
    return TaskData(
        task_name="test-task",
        trial_name="test-task__abc123",
        reward=0.0,
        n_input_tokens=1000,
        n_output_tokens=500,
        n_steps=2,
        trajectory_steps=sample_trajectory_steps,
        test_results=sample_test_results,
        task_prompt="Complete the test task",
    )


class TestCompressTrajectory:
    """Tests for compress_trajectory function."""

    def test_compress_basic(self, sample_trajectory_steps):
        result = compress_trajectory(sample_trajectory_steps)
        assert "[Step 1]" in result
        assert "Starting analysis" in result
        assert "shell:" in result

    def test_compress_limits_steps(self):
        steps = [
            {"step_id": i, "source": "agent", "message": f"Step {i}"}
            for i in range(100)
        ]
        result = compress_trajectory(steps, max_steps=10)
        assert "[Step 10]" in result
        assert "more steps" in result
        assert "[Step 50]" not in result

    def test_compress_empty(self):
        result = compress_trajectory([])
        assert result == ""


class TestFormatTestResults:
    """Tests for format_test_results function."""

    def test_format_with_failures(self, sample_test_results):
        result = format_test_results(sample_test_results)
        assert "2 total" in result
        assert "1 passed" in result
        assert "1 failed" in result
        assert "test_two" in result
        assert "AssertionError" in result

    def test_format_empty(self):
        result = format_test_results({})
        assert "0 total" in result


class TestExtractAmenability:
    """Tests for extract_amenability function."""

    def test_extract_high(self):
        summary = "### Skill Amenability\nHIGH - This task would benefit"
        assert extract_amenability(summary) == "HIGH"

    def test_extract_medium(self):
        summary = "### Skill Amenability\nMEDIUM - Some benefit possible"
        assert extract_amenability(summary) == "MEDIUM"

    def test_extract_low_default(self):
        summary = "No amenability section here"
        assert extract_amenability(summary) == "LOW"


class TestExtractOneLiner:
    """Tests for extract_one_liner function."""

    def test_extract_first_line(self):
        summary = "This is the first line.\n### Header\nMore content"
        assert "This is the first line" in extract_one_liner(summary)

    def test_skip_headers(self):
        summary = "### Header\nActual content here"
        assert extract_one_liner(summary) == "Actual content here"

    def test_empty_summary(self):
        assert "No description" in extract_one_liner("")


class TestLoadTaskData:
    """Tests for load_task_data function."""

    def test_load_valid_task(self, tmp_path):
        task_dir = tmp_path / "test-task__abc123"
        task_dir.mkdir()
        (task_dir / "agent").mkdir()
        (task_dir / "verifier").mkdir()

        result_data = {
            "task_name": "test-task",
            "trial_name": "test-task__abc123",
            "agent_result": {"n_input_tokens": 100, "n_output_tokens": 50},
            "verifier_result": {"rewards": {"reward": 1.0}},
        }
        (task_dir / "result.json").write_text(json.dumps(result_data))

        trajectory_data = {
            "steps": [
                {"step_id": 1, "source": "user", "message": "Task prompt"},
                {"step_id": 3, "source": "user", "message": "Do something"},
            ]
        }
        (task_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory_data))

        ctrf_data = {"results": {"summary": {"tests": 1, "passed": 1, "failed": 0}}}
        (task_dir / "verifier" / "ctrf.json").write_text(json.dumps(ctrf_data))

        data = load_task_data(task_dir)
        assert data is not None
        assert data.task_name == "test-task"
        assert data.reward == 1.0
        assert data.task_prompt == "Do something"

    def test_load_missing_result(self, tmp_path):
        task_dir = tmp_path / "test-task__abc123"
        task_dir.mkdir()
        assert load_task_data(task_dir) is None


class TestWriteTaskSummary:
    """Tests for write_task_summary function."""

    def test_write_summary(self, tmp_path, sample_task_data):
        write_task_summary(tmp_path, sample_task_data, "Test summary content")

        output_file = tmp_path / "task-summaries" / "test-task.md"
        assert output_file.exists()

        content = output_file.read_text()
        assert "# Task: test-task" in content
        assert "FAILED" in content
        assert "Test summary content" in content


class TestGenerateAggregateReport:
    """Tests for generate_aggregate_report function."""

    def test_generate_report(self, tmp_path, sample_task_data):
        summary = "### Skill Amenability\nHIGH - Would help"
        task_summaries = [(sample_task_data, summary)]

        generate_aggregate_report(tmp_path, task_summaries)

        output_file = tmp_path / "summary.md"
        assert output_file.exists()

        content = output_file.read_text()
        assert "# Trajectory Analysis Summary" in content
        assert "Total tasks**: 1" in content
        assert "HIGH" in content


class TestListTaskDirs:
    """Tests for list_task_dirs function."""

    def test_list_valid_dirs(self, tmp_path):
        task1 = tmp_path / "task-one__abc"
        task1.mkdir()
        (task1 / "result.json").write_text("{}")

        task2 = tmp_path / "task-two__def"
        task2.mkdir()
        (task2 / "result.json").write_text("{}")

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "invalid_dir").mkdir()

        dirs = list_task_dirs(tmp_path)
        assert len(dirs) == 2
        assert all("__" in d.name for d in dirs)

    def test_list_empty_dir(self, tmp_path):
        dirs = list_task_dirs(tmp_path)
        assert dirs == []
