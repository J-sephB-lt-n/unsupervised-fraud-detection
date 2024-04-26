# unsupervised-fraud-detection
This repo is still massively under construction

Simulate transaction data:
```bash
$ poetry run python -m data.simdata
```

Create model training dataset:
```bash
$ poetry run python -m feature_eng.create_train_data
```

Run unsupervised anomaly detection models:
```bash
$ poetry run python -m models.dist_to_src_clust_median
$ poetry run python -m models.local_outlier_factor
$ poetry run python -m models.isolation_forest
```

Evaluate models:
```bash
$ poetry run python -m models.evaluate
```
