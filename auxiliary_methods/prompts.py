import pandas as pd
from auxiliary_methods.vector_db import *


def construct_system_prompt(best_example):
    system_prompt = """
    You are a helpful assistant with vast experience with pandas dataframes.
    You have the following dataframes (the description of dataframes will be delimited by '>>><<<' at both ends):
    >>><<<\n"""
    for rt in best_example['Relevant_Tables']:
        system_prompt += TABLE_DESCRIPTIONS[rt]
    system_prompt += '\n>>><<<\n'
    system_prompt += """You also have tools, which you access by outputting JSON structures. 
    Each JSON will include the 'action' key which refers to which tool you use, the parameters the tool requires and a Boolean "solved" flag. 
    If the "solved" flag is true, that mean this JSON is the last one in the process and no more JSONs will be created after. 
    The following are the tools you have for this task (the description of the tools will be delimited by '=-=-=-=' at both ends.)
    =-=-=-=
    """
    for rf in best_example['Relevant_Functions']:
        system_prompt += TOOL_DESCRIPTIONS[rf]
    system_prompt += "\n=-=-=-=\n"
    system_prompt += """When given a question you are to give a chain of thought consisting of thought, action, observation, thought, action, observation etc, 
    as shown in the following fully-solved example, delimited at both ends by '^^^^^':\n^^^^^"""
    system_prompt += "Question: " + best_example['Question'] + '\n'
    system_prompt += "Solution: \n" + best_example['Solution'] + '\n^^^^^\n'
    system_prompt += "KEEP IN MIND THAT THE SOLUTION CANNOT BE TERMINATED IF THE 'solved' FLAG IS False OR THE FINAL ACTION WASN'T EITHER count OR get_value!!!"
    return system_prompt




EXAMPLES = [{"Question":"How many movies were released in 2010?",
             "Solution": """Thought: we need to filter title_basics on movies and the year 2010
Action:{'action': 'filter', 'dataframe': 'title_basics',  'condition': 'titleType=="movie" and startYear=2010', 'solved': False}
Observation: The dataframe title_basics now only have rows for movies from 2010
Thought: We need to know how many movies are in the list, and that is the answer to the question
Action: {'action': 'count', 'dataframe': 'title_basics', 'solved': True}""",
"Relevant_Functions": ['filter', 'count'],
"Relevant_Tables": ['title_basics']},

{"Question": "How many actors were born in 2001?",
 "Solution": """
Thought: First we need to find a list of actors born in 2001
Action: {'action': 'filter', 'dataframe': 'name_basics', 'condition': 'primaryProfession.str.contains("actor") and birthYear==2001', 'solved': False}
Observation: We now have a dataframe of actors born in 2001
Thought: We now need to countthe number of rows in our dataframe
Action: {'action': 'count', 'dataframe': 'name_basics', 'solved': True}
Observation: we now have our answer

""",
"Relevant_Functions": ['filter', 'count'],
"Relevant_Tables": ['name_basics']
},

{"Question": "What is the language of the movie 'Pulp Fiction'?",
"Solution": """Thought: We need to find a movie called Pulp Fiction that is the original title and get the language.
Action: {'action': 'filter', 'dataframe': 'title_akas', 'condition': 'title=="Pulp Fiction" and isOriginalTitle==1', 'solved': False}"
Observation: "we now have the single-row dataframe for the original Pulp Fiction title"
Thought: We need to extract the language from this dataframe and that is the final answer
Action: {'action': 'get_value', 'dataframe': 'title_akas', 'column': 'language', 'solved': True}
Observation: We now have our answer""",
"Relevant_Functions": ['filter', 'get_value'],
"Relevant_Tables": ['title_akas']},

{"Question": "What is the series with the most episodes?",
"Solution": """Thought: We need to find the highest episode number.
Action:{'action': 'filter', 'dataframe': 'title_episodes', 'condition': 'episodeNumber==episodeNumber.max()', 'solved': False}
Observation: We now have a dataframe with the highest value of episodeNumber
Thought: We need to extract the title of the show from the dataframe, but since we only have the parentTconst value of the show, we'll join with a table that has the title"
Action: {'action': 'join', 'left': 'title_episodes', 'right': 'title_basics', 'left_on': 'parentTconst', 'right_on': 'tconst', 'solved': False}
Observation: we now have a dataframe where every row has the most episodes and 
""",
"Relevant_Functions": ['filter', 'join'],
"Relevant_Tables": ['title_episode', 'title_basics']
}

]


TABLE_DESCRIPTIONS = {

"title_basics": 
"""'title_basics' - basic information about titles. Its columns include: 
tconst(unique identifier for each movie on IMDB), 
titleType(movie, short, tvseries, tvepisode, video...), 
startYear: INTEGER year when title was first released
originalTitle (the original name of the movie).""",

"title_akas":
"""'title_akas' - regional information about each title. Its columns include: 
titleId (same as tconst in previous dataframe), 
region, 
title (local title), 
language, 
isOriginalTitle (Boolean whether it is the original title or a local release, which may mean different language from the original. '1' means original, '0' means local).
""",

"name_basics":
"""'name_basics - basic information about people. Columns include:
'nconst': unique string for an individual,
'primaryName' - string of the name the person actually goes by,
birthYear - year of birth (integer),
deathYear - year of death (integer),
primaryProfession - a string with at most 3 profession (actor, soundtrack, producer, writer...),
knownForTitles - a group of tconst values (the primary key of the title_basics dataframe) each representing a title  the person is known for""",

"title_episode":
"""'title_episode - a dataframe of episode. Columns include:
tconst - a unique string for each episode, 
parentTconst - the identifier of the entire show (many episodes can be from the same show),
seasonNumber - number of the season's show that the episode was on (an integer)"""


}


TOOL_DESCRIPTIONS = {
    "filter":
    """'filter': filter a dataframe IN PLACE according to some condition. For example, to get from 'title_basics' only the rows relating to 'Pulp Fiction'
you would give the following JSON:
{'action': 'filter', 'dataframe': 'title_basics', 'condition': 'originalTitle=="Pulp Fiction"', 'solved': False}
The flag 'solved' is False unless the result of the action is the solution required.
The first action taken with any dataframe should ALWAYS be 'filter' because the unfiltered dataframes are too large! 
filter takes the name of the dataframe and the condition to query on.
""",

"count": 
"""'count': counts the rows in the dataframe. Useful if you've filtered a dataframe and want to know how many rows are left.
To get the number of rows in the dataframe title_basics
and show that it is the final step of the solution to the problem you would give the following JSON:
{'action': 'count', 'dataframe': 'title_basics', 'solved': True}""",

"join": 
"""Creates an inner join between 2 dataframes ("left" and "right") on a certain column of each ("left_on", "right_on"). 
So to join title_episodes on the column 'parentTconst' with title_basics on the column 'tconst' you would form the following JSON:
{'action': 'join', 'left': 'title_episodes', 'right': 'title_basics', 'left_on': 'parentTconst', 'right_on': 'tconst', 'solved': False}""", 

"get_value": 
"""'get_value' returns a value from the first row and a specified column of a dataframe. Is useful if there is only one row or if the same value would be in this column for all rows.
For example: {'action': 'get_value', 'dataframe': 'title_episode', 'column': 'episodeNumber', 'solved': True}"""
}





def create_examples_df(examples):
    df = pd.DataFrame(examples)
    df = add_embeddings(df)
    return df

