# HCIA-AI Machine Learning Lab

Python implementations of five basic machine learning algorithms, based on the
Huawei HCIA-AI V3.5 Machine Learning lab guide. I set it up to run as a plain
Python script, fixed a few problems in the guide's code, and added a K-means
implementation from scratch.

## Experiments

| # | Experiment | What it does |
|---|-----------|--------------|
| 1 | Linear regression (scikit-learn) | Fits a line to house area vs price and predicts a price |
| 2 | Linear regression from scratch | Gradient descent with numpy only, plots the loss curve |
| 3 | Logistic regression | Predicts whether a room will be rented, with a probability |
| 4 | Decision tree | Learns rules from weather data to predict whether to play tennis |
| 5 | K-means clustering | Groups 500 points into clusters, comparing k = 3 and k = 4 |

`kmeans_from_scratch.py` answers the lab's question: how to implement K-means
using only numpy.

## Changes from the original guide

- Made the notebook code run as a script (`print()` and `plt.show()` added).
- Replaced Graphviz/pydotplus with matplotlib's `plot_tree`, so no extra install is needed.
- Corrected the decision tree test sample to `[2, 0, 0, 0]`. The guide's
  `[0, 0, 1, 1]` does not match its own description (sunny, cold, high humidity,
  strong wind) under the alphabetical encoding the code produces.
- Fixed typos from the guide.

## Run it

```
pip install numpy matplotlib scikit-learn pandas
python ml_lab.py
python kmeans_from_scratch.py
```

Run from the project folder so the data paths (`ML/tennis.txt`,
`ML/02/lr2_data.txt`) resolve. Close each graph window to continue.

## Credits

Lab content and datasets: Huawei HCIA-AI V3.5 training materials.
