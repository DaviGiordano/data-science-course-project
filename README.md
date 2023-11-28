# data-science-course-project
This is a data science course project with the goal to understand the impact of the different choices made along the data science process in classification and forecasting model implementations

### To run the notebooks for your specific dataset:
In the first cell, when appropriate, define these variables:
```
filename = "relative_path_for_csv_file"
file_tag = "tag_for_filenaming"
target = "target_value_of_dataset"
```

## Plan

### Data Profiling

Characterize data along the four perspectives: 
- Dimensionality
  - #records vs. #variables
  - #variables per type
  - #missing values for each variable
- Distribution
  - Global boxplots for numerical variables
  - Single variable boxplots for numerical variables
  - Histograms for numerical variables
  - Outliers study for numerical variables
  - **What about non-numerical variables? Maybe bar charts?**
  - Class distribution
- Granularity
  - Granularity study for location
  - Granularity study for payment_behavior
  - **Should we include this in the dataset, e.g. region as a new feature? What else would be the value of this?**
- Sparsity
  - Sparsity analysis (Scatter plots)
    - **what about non-numerical variables?**
  - Correlation analysis (heatmap)
    - **how did we encode symbolic variables?**
    - "_When in the presence of symbolic variables [...] students have to choose a new encoding for those variables, before proceeding with the correlation analysis_"

- **Generally, what insights did we gain from the data profiling?**

### Data Preparation

- Variables encoding
- **Should we split a held-out-test set before the missing value imputation?**

- Missing Value Imputation
  - **How did we do the missing values imputation? Did we remove any variables with too many missing values? If so, what was the threshold? Also, what did we do to the remaining rows with missing columns? Did we just drop those or did we replace the values somehow?**
  - Results: NB & KNN for two approaches
    - **how did we evaluate these approaches? for exmample, which hparams?**
    - **If we replaced the missing values somehow, the value to replace with should be based on a training set and the evaluation should be conducted on another validatoin set**
- Outliers Treatment
  - test two different approaches
  - if strategy is based on data (median) use the median of a training set and evaluate using a validation set
  - evaluate with NB and KNN (**which hyperparameters?**)
  - finally, use the whole training set to apply the best-performing approach
- Scaling
  - **only scale original numerical features**
  - test min-max scaling, z-score 
  - if strategy is based on data (median) use the median of a training set and evaluate using a validation set
  - evaluate with KNN (**which hyperparameters?**)
  - finally, use the whole training set to apply the best-performing approach
- Balancing
  - study undersampling, oversampling and SMOTE 
  - if strategy is based on data (median) use the median of a training set and evaluate using a validation set
  - evaluate with NB and KNN (**which hyperparameters?**)
  - finally, use the whole training set to apply the best-performing approach


### Model's Evaluation

- **Which model alternatives do we test and which hyperparameters do we tune?**
- **How do we conduct the hyperparameter search: Cross-val or using a validation set?**
- **What metrics do we examine and which metric do we use to choose the best model?**
- **How does the overfitting analysis look like?**