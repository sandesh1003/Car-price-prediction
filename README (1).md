# Task 3: Car Price Prediction with Machine Learning

A regression pipeline that predicts used-car prices from features such as **brand goodwill, horsepower and mileage**, built with Pandas, Scikit-learn and Matplotlib.

## Run

```bash
pip install pandas scikit-learn matplotlib joblib
python car_price_prediction.py                                   # built-in synthetic dataset
python car_price_prediction.py --data my_cars.csv --target price # your own CSV
```

Everything is written to `outputs/`: the dataset, `model_comparison.csv`, seven plots and `car_price_model.joblib`.

> The built-in dataset is **synthetic** (prices come from a made-up formula plus noise), so the scores show that the pipeline works, not how well real car prices can be predicted. Use `--data` with a real CSV for real results.

## What each step does

| Task requirement | Where it happens |
|---|---|
| Collect car features (brand goodwill, horsepower, mileage, ...) | `generate_data()` or `--data` |
| Data preprocessing | `clean()` removes duplicates and impossible values; the pipeline imputes missing values (median / most frequent), scales numbers and one-hot encodes text columns |
| Feature engineering | `add_features()`: car age, km per year, log(km), horsepower per litre, goodwill x power |
| Train a regression model | Linear Regression, Ridge, Random Forest, Gradient Boosting and a grid-searched Gradient Boosting; price is modelled on a log scale |
| Model evaluation | 5-fold cross-validation on the training set, then MAE, RMSE and R2 on a held-out 20% test set; the final model is chosen by CV only |
| Pandas, Scikit-learn, Matplotlib | data handling, pipelines and models, and all plots |

## Plots in `outputs/`

1. `01_price_distribution.png` - raw and log price
2. `02_correlation_heatmap.png`
3. `03_price_vs_features.png`
4. `04_model_comparison.png`
5. `05_actual_vs_predicted.png`
6. `06_residuals.png`
7. `07_feature_importance.png`

## Predict for a new car with the saved model

```python
import joblib, pandas as pd
from car_price_prediction import add_features  # needed to load the saved pipeline

model = joblib.load("outputs/car_price_model.joblib")
car = pd.DataFrame([{
    "brand": "Hyundai", "brand_goodwill": 7.6, "year": 2020, "horsepower": 100,
    "engine_cc": 1200, "mileage_kmpl": 20, "km_driven": 45000, "fuel_type": "Petrol",
    "transmission": "Manual", "owners": 1, "seats": 5,
}])
print(model.predict(car))  # price in INR lakh
```

## Real-world applications of ML price prediction

- **Used-car marketplaces and dealers** - suggest a fair listing price and flag listings that are over- or under-priced.
- **Insurance and lending** - estimate a car's current value for claims, loans and trade-ins.
- **Buyers and sellers** - check a quote against what similar cars sold for.
- **Fleet and leasing companies** - forecast resale value and depreciation when planning purchases.
- The same approach (features in, price out) is used for house prices, flights, hotels and e-commerce pricing.

## Limits to keep in mind

- Predictions are only as good as the data: a model trained on one market or period can be wrong in another.
- Features such as condition, accident history and local demand are often missing but affect price a lot.
- Compare models on cross-validation, not on the test set, and keep the test set for one final check.
