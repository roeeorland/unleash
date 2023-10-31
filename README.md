# unleash

## Data Loading and Cleaning

Due to the size of the .tsv files, the dataframes are read in chunks , and are filtered at the time of first loading (we assume a dataframe filtered on some condition is small enough).
Missing data is presented as '\N'. This means that columns with missing values cannot be of type int. We replace all occurrences of '\N' with 0 and then force columns to type int (with a try...except clause, meaning that only numerical columns are converted. This step is important because when we query the dataframes there is a difference between 'deathYear==1984' and 'deathYear=="1984"'

## Running the code

### Datasets 
The source .tsv files were not uploaded to github. They can be downloaded at https://drive.google.com/drive/folders/1eEw9m3U1XAaONxF2-SGDy7D9UgonYw7o?usp=sharing
The files should be placed under the datasets subfolder.

### Virtual Environment
Please activate a local virtual environment and install the requirements (pip install -r requirements.txt).

### Openai access

In the vector_db.py file I load my openai api key. Feel free to use the method I did, or just paste the key, or using an environment variable.


