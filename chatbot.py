import pandas as pd
import json
import openai
import re
from prompts.prompts import *
from data_cleaning import *


database_dict = {"name_basics": {"data":None,
                                  "isLoaded": False},
                  "title_akas": {"data":None,
                                  "isLoaded": False},
                    "title_basics": {"data":None,
                                  "isLoaded": False},
                  "title_crew": {"data":None,
                                  "isLoaded": False},
                   "title_episode": {"data":None,
                                  "isLoaded": False},
                  "title_principals": {"data":None,
                                  "isLoaded": False},
                  "title_ratings": {"data":None,
                                  "isLoaded": False},
                                  }





class BIBot:
    def __init__(self):
        with open('/home/roee/.credentials/roee_credentials.json', 'r') as fp:
            self.openai_api_key = json.load(fp)['OPENAI_API_KEY']
        self.system_prompt = system_prompt
        self.function_dict = {'filter': self.filter, 'count': self.count}
        self.tables_dict = {"name_basics": {"data":None,
                                  "isLoaded": False},
                  "title_akas": {"data":None,
                                  "isLoaded": False},
                    "title_basics": {"data":None,
                                  "isLoaded": False},
                  "title_crew": {"data":None,
                                  "isLoaded": False},
                   "title_episode": {"data":None,
                                  "isLoaded": False},
                  "title_principals": {"data":None,
                                  "isLoaded": False},
                  "title_ratings": {"data":None,
                                  "isLoaded": False},
                                  }
    
    
    
    
    def api_call(self, question, example):
        openai.api_key = self.openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt + '\n' + "example:\n" + "Question: How many movies were released in 2010?\nSolution:\n" + example},
                {"role": "user", "content": question}],
            temperature=0)
        self.result_string = response['choices'][0]['message']['content']
        print(self.result_string)

    def parse_api_call(self):
        self.actions = re.findall("\{[^\}]+\}", self.result_string)

    
    ## Functions
    def filter(self, df_name, condition, chunk_size=1000):

        if not self.tables_dict[df_name]["isLoaded"]:
            mini_frames = []
            for chunk in pd.read_csv('datasets/' + df_name + '.csv', delimiter='\t', chunksize=chunk_size):
                chunk = chunk.query(condition, engine='python')
                if chunk.shape[0]:
                    mini_frames.append(chunk)
            result_df = pd.concat(mini_frames, ignore_index=True)
            result_df.replace('\N', 0, inplace=True)
            result_df = fix_int_type(result_df)
            self.tables_dict[df_name]["data"] = result_df
            self.tables_dict[df_name]["isLoaded"] = True
            
        else:
            df = self.tables_dict[df_name]["data"]
            result_df = df.query(condition)
            self.tables_dict[df_name]["data"] = result_df
        


    def count(self, df_name):
        return self.tables_dict[df_name]["data"].shape[0]
    


test_instance = BIBot()
test_instance.api_call("how many shorts were released in 2007?", """Thought: we need to filter on movies and the year 2010
Action:{'action': 'filter', 'dataframe': 'title_basics',  'condition': 'titleType=="movie" and startYear=2010', 'solved': False}
Observation: The dataframe title_basics now only have rows for movies from 2010
Thought: We need to know how many movies are in the list, and that is the answer to the question
Action: {'action': 'count', 'dataframe': 'title_basics', 'solved': True}""")