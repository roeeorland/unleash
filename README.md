# unleash

## Data Loading and Cleaning

Due to the size of the .tsv files, the dataframes are read in chunks , and are filtered at the time of first loading (we assume a dataframe filtered on some condition is small enough).
Missing data is presented as '\N'. This means that columns with missing values cannot be of type int. We replace all occurrences of '\N' with 0 and then force columns to type int (with a try...except clause, meaning that only numerical columns are converted. This step is important because when we query the dataframes there is a difference between 'deathYear==1984' and 'deathYear=="1984"' 


