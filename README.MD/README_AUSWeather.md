# Australian Weather Prediction | Machine Learning

## Project overview

This project develops and evaluates machine learning classification models to predict whether it will rain today in the Melbourne area, Australia, using historical meteorological observations.

The work was completed as a final project for the IBM/Coursera Machine Learning course. It demonstrates an end-to-end supervised learning workflow, from data preparation and feature engineering to model optimization, evaluation, and interpretation.

## Business / analytical question

Can historical weather conditions help predict whether it will rain today in the Melbourne area?

A reliable rainfall prediction can support everyday planning and decisions that depend on weather conditions. This notebook is an educational modeling exercise and is not a production weather-forecasting service.

## Dataset

- **Source:** Australian Government Bureau of Meteorology data, distributed through the course's Skills Network dataset.
- **Dataset used in the notebook:** `weatherAUS-2.csv`
- **Geographic scope:** Melbourne, Melbourne Airport, and Watsonia.
- **Target variable:** `RainToday` (after the notebook's target-column renaming).
- **Data preparation:** rows with missing values are removed; the `Date` column is converted into a seasonal feature and then removed.

The notebook first renames `RainTomorrow` to `RainToday` and `RainToday` to `RainYesterday`. It then uses the renamed `RainToday` as the target. This reframes the task as predicting today's rain from the available weather observations.

## Workflow

1. **Load and inspect the data** with Pandas.
2. **Clean the dataset** by removing rows with missing values.
3. **Address prediction timing / leakage considerations** by redefining the rainfall target.
4. **Limit the geographic scope** to Melbourne, Melbourne Airport, and Watsonia.
5. **Engineer a seasonal feature** from the date.
6. **Separate features and target** and inspect class balance.
7. **Split the data** into training and test sets (80/20), using stratification.
8. **Build preprocessing pipelines:**
   - StandardScaler for numerical variables.
   - OneHotEncoder for categorical variables.
   - ColumnTransformer to apply the appropriate transformation to each feature type.
9. **Train and tune a Random Forest classifier** with GridSearchCV and cross-validation.
10. **Evaluate the model** using accuracy, a classification report, and a confusion matrix.
11. **Inspect feature importance** from the fitted Random Forest.
12. **Train and evaluate Logistic Regression** and compare its results with Random Forest.

## Models and results

The following values are taken from the saved notebook outputs:

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| Test accuracy | 84% | 83% |
| Recall — `Yes` (rain) | 51% | 51% |
| Cross-validation score | 85% | Not reported as a final comparison value |

### Interpretation

The models have similar overall accuracy and similar recall for rainy days. However, a recall of approximately 51% for the `Yes` class means that the models identify only about half of the actual rainy days in the test set.

Accuracy alone is therefore not sufficient to assess this problem. Because the target classes are imbalanced (the notebook reports approximately 78% `No` and 22% `Yes`), recall, precision, F1-score, and the confusion matrix are important for understanding the model's practical limitations.

## Tools and libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

Scikit-learn components used include `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, `GridSearchCV`, `StratifiedKFold`, `RandomForestClassifier`, `LogisticRegression`, and classification metrics.




## Key learnings

- Preparing real-world weather data for supervised learning.
- Understanding class imbalance and its impact on model evaluation.
- Building reproducible preprocessing and modeling pipelines.
- Tuning hyperparameters with cross-validation.
- Comparing tree-based and linear classification models.
- Interpreting precision, recall, F1-score, accuracy, and confusion matrices.
- Communicating model limitations instead of relying on accuracy alone.

## Limitations and future improvements

- The model is limited to three locations around Melbourne; results should not be generalized to all of Australia without further validation.
- Removing all rows with missing values may discard useful information. Imputation strategies could be tested.
- Rainfall is the minority class, and the current recall indicates room for improvement.
- Future work could test class weighting, threshold adjustment, additional feature engineering, and other classification algorithms.
- A time-based validation strategy would be useful to test performance on later, unseen periods and reduce the risk of overly optimistic random-split results.
- The model should be evaluated with the exact information that would be available at prediction time before any operational use.

## Author

**Julian López**  
Economics graduate | Data Analytics | Machine Learning

GitHub: https://github.com/Julianlo11

---

*Academic project developed as part of the IBM/Coursera Machine Learning coursework. Dataset attribution and original data source are described in the notebook.*
