import shutil

import pandas as pd

from wine_analysis import (
    import_dataset,
    inspect_data,
    filter_data,
    group_data,
    train_model,
    plot_boxplot,
    plot_scatter,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", shutil.get_terminal_size(fallback=(120, 24)).columns)

# Step 1: Importing the dataset
wine = import_dataset("data/wine_quality_merged.csv")

print(" " " ")

# Step 2: Inspecting the data
inspect_data(wine)

print(" " " ")

# Step 3: Filtering
filter_data(wine)

print(" " " ")

# Step 4: Grouping
group_data(wine)

print(" " " ")

# Step 5: Machine learning model
train_model(wine)

print(" " " ")

# Step 6: Visualization of Boxplot
plot_boxplot(wine, "graphs/quality_vs_alcohol.png")

# Step 7: Visualization of Scatter Plot with Trend Line
plot_scatter(wine, "graphs/alcohol_vs_density.png")
