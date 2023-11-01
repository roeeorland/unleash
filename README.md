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

### The actual running of the code
From the virtual environment, while in the main forlder: python chatbot.py


## Proposed Architecture

The purpose of the project is to construct a chatbot to answer questions related to a database provided by IMDB

To do this our chatbot reads the question, finds a similar, solved question in the examples provided, and creates a system prompt for a LLM which includes a description of the general problem ("you answer questions using IMDB datasets..."), a description of the tables and tools RELEVANT TO THE PARTICULAR PROBLEM, as well as the worked example.
The LLM uses a ReAct-atyle chain of thought to arrive at a solution. The tools are then used according to the solution to arrive at the appropriate answer.

## The process in detail:

### Constructing the prompts

Dictionaries containing complete examples, as well as description of all tools and tables are located in auxiliary_functions/prompts.py
When the chatbot is launched a ChromaDB collection is launched. All example questions are embedded into a vector space and saved in the collection.
When a question is submitted, the question is also embedded and the algorithm searches for the nearest neighbor among the embeddings of the example questions.
Every solved example in the dictionary also includes lists of the relevant tools and tables. These lists trigger the adding of specific text from dictionaries describing the tables and tools (this is done to minimize the amount of text sent to the LLM and ensure that it stays focused).
The LLM processes the text and return a ReAct chain-of-thought solution (Thought->Action->Observation->Thought->Action->Observation->...). The actions are JSON objects which contain instructions on how to use the tools. These actions are parsed from the string reply.
I refrained from using LangChain because of its requirement that all tools be loaded. Also, their dataframe tools would not have helped as they require loading the entire dataframes. 

### The datasets
The datasets are huge and must be handled locally. For this reason Every dataset that is needed for a particular job is loaded in chunks, and only if it is filtered according to some condition at the same time.
As unavailable data is marked by '\N', when querying an otherwise-numerical column the type is automatically set to string. To fix this we fill all '\N' cells with the '0' char, and then attempt to change the column to int (if it doesn't work it means the column wasn't purely an integer column anyway).
The solution of chunking is very slow, and there is probably a better solution which I have missed (and which does not involve spark) 

