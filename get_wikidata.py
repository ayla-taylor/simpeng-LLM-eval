import wikipediaapi
from collections import defaultdict
import json


def create_wordlists(user_agent:str):
    wiki_simple = wikipediaapi.Wikipedia(user_agent=user_agent, language='simple')
    simple_wordlist = wiki_simple.page('Wikipedia:Basic_English_combined_wordlist')
    word_sections = defaultdict(list)
    for line in simple_wordlist.text.split("\n"):
        section = line.split(":")
        if section[0].lower() in ["basic", "international", "addendum", "compound", "endings"]:
            word_sections[section[0]].append(section[1])
    word_lists = {}
    for section, lists in word_sections.items():
        section_list = []
        for words in lists:
            ascii_list = words.encode(encoding="ascii", errors="replace").decode("ascii").split("?")
            cleaned_words = []
            for word in ascii_list: 
                if '/' in word: 
                    cleaned_words += word.split("/")
                else: 
                    cleaned_words.append(word)
            ascii_list = cleaned_words
            section_list += [word.strip() for word in ascii_list]
        word_lists[section] = section_list
    return word_lists
