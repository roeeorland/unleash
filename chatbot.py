import pandas as pd
import json
import openai
import re
from auxiliary_methods.prompts import *
from auxiliary_methods.data_cleaning import *
from auxiliary_methods.vector_db import *


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
        self.collection = create_embeddings_collection([e["Question"] for e in EXAMPLES])
        while True:
            # self.system_prompt = system_prompt
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
            # self.examples_df = create_examples_df(EXAMPLES)
            self.question = self.get_user_input()
            self.run_pipeline()

    def get_user_input(self):
        user_input = input("What is your question?\n")
        return user_input
    
    
    def run_pipeline(self):

    # try:
        idx, question = get_best_match(self.question, self.collection)
        best_example = EXAMPLES[eval(idx)]
        self.system_prompt = construct_system_prompt(best_example)
        self.api_call()
        self.parse_api_call()
        for action in self.actions:
            func = self.assign_function(action)
            func(action)
        if not self.result:
            print('Could not find the records')
        else:
            print(f"The answer is {self.result}.\n\n")
        print("\n\nCare to ask me another?\n")
    # except:
    #     print("Failed to answer your question.\n\n Care to ask me another?\n")
       
    
    
    def api_call(self):
        openai.api_key = self.openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.question}],
            temperature=0)
        self.api_reply = response['choices'][0]['message']['content']


    def parse_api_call(self):
        actions = re.findall("\{[^\}]+\}", self.api_reply)
        self.actions = [eval(action) for action in actions]


    
    
    ## Functions
    def filter(self, action, chunk_size=10000):

        if not self.tables_dict[action['dataframe']]["isLoaded"]:
            mini_frames = []
            i = 1
            for chunk in pd.read_csv('datasets/' + action['dataframe'] + '.tsv', delimiter='\t', chunksize=chunk_size):
                if not i%100:
                    print(f"Read {i * chunk_size} rows of {action['dataframe']}")
                i += 1
                q = chunk.query(action['condition'], engine='python').shape[0]
                
                chunk.fillna('0', inplace=True)
                chunk.replace('\\N', '0', inplace=True)
                chunk = fix_int_type(chunk)
                
                chunk = chunk.query(action['condition'], engine='python')
               
                if chunk.shape[0]:
                    mini_frames.append(chunk)
               
            if not mini_frames:
                print("Could not find any results in the dataframe")
            result_df = pd.concat(mini_frames, ignore_index=True)
            result_df = result_df.query(action['condition'], engine='python')
            result_df.reset_index(inplace=True, drop=True)
            

            self.tables_dict[action['dataframe']]["data"] = result_df
            self.tables_dict[action['dataframe']]["isLoaded"] = True
            
        else:
            df = self.tables_dict[action['dataframe']]["data"]
            result_df = df.query(action['condition'])
            self.tables_dict[action['dataframe']]["data"] = result_df

        


    def count(self, action):
        self.result = self.tables_dict[action['dataframe']]["data"].shape[0]

    def join(self, action_dict):
        left = self.tables_dict[action_dict['left']]['data'] 
        if not self.tables_dict[action_dict['right']]['isLoaded']:
            # right = pd.read_csv('datasets/' + action_dict['right'] + '.tsv', delimiter='\t')
            self.filter({'dataframe':action_dict['right'], 'condition': f'tconst==\'{left["parentTconst"][0]}\''})
        
        right = self.tables_dict[action_dict['right']]['data']
        self.tables_dict[action_dict['left']]['data'] = left.merge(right, left_on=action_dict['left_on'], right_on=action_dict['right_on'], how='inner')
    
    def get_value(self, action_dict):
        column = action_dict['column']
        self.result = self.tables_dict[action_dict['dataframe']]['data'][column][0]
    
    def get_relevant_filmography(self, action):
        name = action['person']
        self.filter({'dataframe': 'name_basics', 'condition': f'primaryName == \'{name}\''})
        self.get_value({'column': 'nconst', 'dataframe': 'name_basics'})
        self.filter({'dataframe': 'title_principals', 'condition': f"nconst == \'{self.result}\'"})
        self.filter({'dataframe': 'title_basics', 'condition': f'genres.str.contains(\"{action["genre"]}\")'})
        self.join_for_filmography({'left': 'title_principals', 'right': 'title_basics', 'left_on': 'tconst', 'right_on': 'tconst'})

    def join_for_filmography(self, action_dict):
        if not self.tables_dict[action_dict['right']]['isLoaded']:
            right = pd.read_csv('datasets/' + action_dict['right'] + '.tsv', delimiter='\t')
            # self.filter({'dataframe':action_dict['right'], 'condition': f'tconst==\'{left["parentTconst"][0]}\''})
        else:
            right = self.tables_dict[action_dict['right']]['data']
        self.tables_dict[action_dict['left']]['data'] = left.merge(right, left_on=action_dict['left_on'], right_on=action_dict['right_on'], how='inner')
    

        
    def assign_function(self, action):
        return getattr(self, action['action'])
    
    


test_instance = BIBot()
# test_instance.api_call("how many shorts were released in 2007?", """Thought: we need to filter on movies and the year 2010
# Action:{'action': 'filter', 'dataframe': 'title_basics',  'condition': 'titleType=="movie" and startYear=2010', 'solved': False}
# Observation: The dataframe title_basics now only have rows for movies from 2010
# Thought: We need to know how many movies are in the list, and that is the answer to the question
# Action: {'action': 'count', 'dataframe': 'title_basics', 'solved': True}""")