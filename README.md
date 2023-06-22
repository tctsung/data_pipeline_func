# data_pipeline_func
Demo for self-written data cleaning, EDA, ML, DL functions in python &amp; R

0. Define the problem & collect data
1. Overview of the data 
  * goal: understand the str, size, basic info
  * Eg. tabular data ([preprocess.ipynb](https://github.com/tctsung/data_pipeline_func/blob/main/preprocess.ipynb))
2. Data preprocessing 
  * goal: transform to proper dtypes, extract features
  * Eg. tabular data ([preprocess.ipynb](https://github.com/tctsung/data_pipeline_func/blob/main/preprocess.ipynb))
3. Exploratory data analysis 
  * goal: visualize distribution, missingness, relationships betweeen features & with main outcome
  * Eg. tabular data ([EDA.ipynb](https://github.com/tctsung/data_pipeline_func/blob/main/EDA.ipynb))
4. Baseline Model
  * goal: see if the problem is feasible & evaluation criterion is reasonable
5. Model optimization
  * goal: select the proper method
6. Comprehensive model evaluation:
  * goal: have a comprehensive view and find data insights
8. Model Deployment
  * goal: automate previous process & build a end-to-end data pipeline
9. Model maintenance, monitoring

## Notebooks

### preprocess.ipynb
This notebook includes the step 1 & 2 in a ML data pipeline mainly using Pandas & self-written OOP [data_cleaner.py](https://github.com/tctsung/data_pipeline_func/blob/main/funcs/data_cleaner.py)

### EDA.ipynb

This notebook includes the exploratory data analysis for tabular data mainly using seaborn, matplotlib & self-written OOP [data_visualizer.py](https://github.com/tctsung/data_pipeline_func/blob/main/funcs/data_visualizer.py)

## funcs

This folder contains helper class & functions for the data pipeline

### [data_cleaner.py](https://github.com/tctsung/data_pipeline_func/blob/main/funcs/data_cleaner.py)

Helper for data preprocessing & information extraction in tabular data

* class Cleaner: an class that do a whole pipeline of inplace operation for input data frame
* function overview: Check the data types & NA percentage & unique values of the data
* function match_cols: return the column names using regex to filter
* function extract_date_features: extract extra features from date 
* function standardization: supports normalization & min max standardization for numeric features
* function label_encoding: turn categorial features into Int64 while keeping the original labels in another DF

## Data

### raw_data

Folder with raw data
* ahp.csv (source: R CRAN package [r02pro](https://r02pro.github.io/)

### clean_data

The folder with processed data

* ahp.parquet.gzip: processed ahp.csv data using [preprocess.ipynb](https://github.com/tctsung/data_pipeline_func/blob/main/preprocess.ipynb) 

