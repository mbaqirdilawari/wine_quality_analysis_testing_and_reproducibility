"""
Unit + system tests for wine_analysis.py.

Run with:
    pytest -v
"""

import os

import pandas as pd
import pytest

from wine_analysis import (
    import_dataset,
    inspect_data,
    filter_data,
    group_data,
    train_model,
    plot_boxplot,
    plot_scatter,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "wine_quality_merged.csv")

# Loads the real dataset once, shared by every test below
@pytest.fixture(scope="module")
def wine_df():
    return import_dataset(DATA_PATH)


# Data loading test: checks the CSV loads into the right shape
def test_import_dataset_returns_expected_shape(wine_df):
    assert isinstance(wine_df, pd.DataFrame)
    assert wine_df.shape == (6497, 13)


# Data loading test (edge case): a missing file should raise an error, not fail silently
def test_import_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        import_dataset("data/this_file_does_not_exist.csv")


# Preprocessing test: checks the known duplicate-row count is detected correctly
def test_inspect_data_finds_known_duplicate_count(wine_df):
    duplicate_count = inspect_data(wine_df)
    assert duplicate_count == 1177


# Preprocessing/transformation test: checks each of the 5 filters returns only matching rows
def test_filter_data_splits_correctly(wine_df):
    high_quality, bad_quality, medium_quality, high_alcohol_red, low_alcohol_red = (
        filter_data(wine_df)
    )

    assert (high_quality["quality"] >= 7).all()
    assert (bad_quality["quality"] <= 4).all()
    assert (medium_quality["quality"] > 4).all() and (
        medium_quality["quality"] < 7
    ).all()
    assert (high_alcohol_red["type"] == "red").all() and (
        high_alcohol_red["alcohol"] > 12
    ).all()
    assert (low_alcohol_red["type"] == "red").all() and (
        low_alcohol_red["alcohol"] < 10
    ).all()

    total = len(high_quality) + len(bad_quality) + len(medium_quality)
    assert total == len(wine_df)


# Preprocessing/transformation test: checks grouping by type and by quality both work
def test_group_data_groups_by_type_and_quality(wine_df):
    by_type, by_quality = group_data(wine_df)
    assert set(by_type.index) == {"red", "white"}
    assert by_type.loc["white", "no_of_wines"] > by_type.loc["red", "no_of_wines"]
    assert len(by_quality) > 0


# ML model test: checks the model trains and produces sane metrics
def test_train_model_returns_fitted_model_and_metrics(wine_df):
    model, mse, r2 = train_model(wine_df)
    assert hasattr(model, "coef_")
    assert len(model.coef_) == 3
    assert mse > 0
    assert -1.0 <= r2 <= 1.0


# Visualization test: checks the boxplot actually saves a real image file
def test_plot_boxplot_creates_file(wine_df, tmp_path):
    save_path = tmp_path / "quality_vs_alcohol_test.png"
    plot_boxplot(wine_df, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


# Visualization test: checks the scatter plot actually saves a real image file
def test_plot_scatter_creates_file(wine_df, tmp_path):
    save_path = tmp_path / "alcohol_vs_density_test.png"
    plot_scatter(wine_df, str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


# System test: runs every step back-to-back, exactly like analysis.py does
def test_end_to_end_pipeline_runs_without_error(tmp_path):
    wine = import_dataset(DATA_PATH)
    inspect_data(wine)
    filter_data(wine)
    group_data(wine)
    model, mse, r2 = train_model(wine)
    assert mse > 0

    plot_path = tmp_path / "system_test_plot.png"
    plot_boxplot(wine, str(plot_path))
    assert plot_path.exists()
