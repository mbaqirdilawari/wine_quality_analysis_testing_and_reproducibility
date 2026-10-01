"""
Testable functions, for the Wine Quality analysis pipeline.

Every step from the README walkthrough lives here as its own function.
analysis.py calls these in order; test_wine_analysis.py calls them
individually to check each step works on its own.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

matplotlib.use("Agg")  # lets charts save without needing an actual screen
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

FEATURES = ["alcohol", "volatile acidity", "sulphates"]

# Thresholds used to filter wines in Step 3
HIGH_QUALITY_MIN = 7
LOW_QUALITY_MAX = 4
HIGH_ALCOHOL_MIN = 12
LOW_ALCOHOL_MAX = 10

# Columns shown when previewing each filtered group of wines
PREVIEW_COLUMNS = ["type", "alcohol", "quality"]

# Alcohol statistics shared by both groupings in Step 4
ALCOHOL_STATS = {
    "avg_alcohol": ("alcohol", "mean"),
    "std_alcohol": ("alcohol", "std"),
    "min_alcohol": ("alcohol", "min"),
    "max_alcohol": ("alcohol", "max"),
}


def import_dataset(path):
    # Step 1: Importing the dataset
    wine = pd.read_csv(path)
    print(f"Total rows: {len(wine)}")
    print(wine["type"].value_counts())
    return wine


def inspect_data(wine):
    # Step 2: Inspecting the data
    print(f"Shape (rows, columns): {wine.shape}")
    print(wine.head())
    wine.info()
    print(wine.describe())
    print("Missing values:")
    print(wine.isnull().sum())
    duplicate_count = wine.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")
    return duplicate_count


def apply_filter(wine, condition, label):
    """Keep only the rows matching `condition`, then print a short preview."""
    subset = wine.query(condition)
    print(f"{label}: {len(subset)} out of {len(wine)}")
    print(subset[PREVIEW_COLUMNS].head())
    return subset


def filter_data(wine):
    # Step 3: Filtering
    high_quality = apply_filter(
        wine, f"quality >= {HIGH_QUALITY_MIN}", "High quality wines"
    )
    low_quality = apply_filter(
        wine, f"quality <= {LOW_QUALITY_MAX}", "Bad quality wines"
    )
    medium_quality = apply_filter(
        wine,
        f"quality > {LOW_QUALITY_MAX} and quality < {HIGH_QUALITY_MIN}",
        "Medium quality wines",
    )
    high_alcohol_red = apply_filter(
        wine,
        f"type == 'red' and alcohol > {HIGH_ALCOHOL_MIN}",
        "High alcohol red wines",
    )
    low_alcohol_red = apply_filter(
        wine,
        f"type == 'red' and alcohol < {LOW_ALCOHOL_MAX}",
        "Low alcohol red wines",
    )
    return high_quality, low_quality, medium_quality, high_alcohol_red, low_alcohol_red


def group_data(wine):
    # Step 4: Grouping
    by_type = wine.groupby("type").agg(
        **ALCOHOL_STATS,
        avg_quality=("quality", "mean"),
        no_of_wines=("quality", "count"),
    )
    print(by_type)

    by_quality = wine.groupby("quality").agg(
        **ALCOHOL_STATS,
        no_of_wines=("alcohol", "count"),
    )
    print(by_quality)

    return by_type, by_quality


def train_model(wine):
    # Step 5: Machine learning model
    print("Machine Learning Model: Predicting Wine Quality")

    X = wine[FEATURES]
    y = wine["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse:.3f}")
    print(f"R-squared: {r2:.3f}")

    # Print each feature's learned coefficient
    for feature, coef in zip(FEATURES, model.coef_):
        print(f"{feature}: {coef:.3f}")

    return model, mse, r2


def save_plot(title, xlabel, ylabel, save_path):
    """Add a title and axis labels to the current chart, save it, then close it."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_boxplot(wine, save_path):
    # Step 6: Visualization of Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=wine,
        x="quality",
        y="alcohol",
        hue="type",
        palette={"red": "firebrick", "white": "wheat"},
    )
    save_plot(
        "Alcohol Content by Wine Quality Score",
        "Quality Score",
        "Alcohol (%)",
        save_path,
    )
    print(f"Boxplot is saved as {save_path}")
    return save_path


def plot_scatter(wine, save_path):
    # Step 7: Visualization of Scatter Plot with Trend Line
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=wine,
        x="alcohol",
        y="density",
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"},
    )
    save_plot(
        "Alcohol Content vs. Density (All Wines)", "Alcohol (%)", "Density", save_path
    )
    print(f"Scatter plot with trend line saved as {save_path}")
    return save_path
