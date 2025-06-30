import spacy
import json


nlp = spacy.load("en_core_web_sm")


class Analysis:
    
    def __init__(self, text):
        self.full_text = text
        self.doc = nlp(self.full_text)
        self.sents = self.doc.sents
        self.basic_word_count = 0
        self.word_lists = self.load_word_lists()
        self.results_data = {}
        
        
    def load_word_lists(self): 
        with open("word_lists.json", "r") as file: 
            data = json.load(file)
        return data
        
    def get_basic_rate(self, word_list = "Basic"):
        total_len = len(self.doc)
        for token in self.doc: 
            if token.lemma_ in self.word_lists[word_list]: 
                self.basic_word_count += 1
        basic_word_rate = self.basic_word_count/total_len
        return basic_word_rate
       