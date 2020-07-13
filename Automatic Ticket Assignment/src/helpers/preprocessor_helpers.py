# import requests
import os
import re

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
# %matplotlib inline

#Importing libraries for text pre-processing
import spacy
import nltk
nltk.download('stopwords')
from nltk.tokenize.toktok import ToktokTokenizer
#from nltk.tokenize import word_tokenize
from contractions import contractions_dict
import unicodedata
from typing import Dict, List

class PreprocessorHelper:
    def __init__(
        self,
        regex_usernames_to_remove=r"",
        regex_terms_to_remove=r"",
        specific_stopwords=[],
    ):
        self.nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
        self.tokenizer = ToktokTokenizer()
        self.stopword_list = nltk.corpus.stopwords.words('english')
        self.extend_stopword_list(specific_stopwords)
        self.regex_terms_to_remove = regex_terms_to_remove
        self.regex_usernames_to_remove = regex_usernames_to_remove

    def extend_stopword_list(self, extend_with: List[str]):
        non_stopwords = ["no", "not"]
        for word in non_stopwords:
            try:
                self.stopword_list.remove(word)
            except ValueError:
                # continue of given word is already removed from stopword list
                continue

        self.stopword_list = list(set(self.stopword_list + extend_with))

    # remove name initials
    def get_name_initials(self, name: str) -> str:
        l = []
        for i in name.split(' '):
            l.append(i[0])
        initials = ''.join(l)
        return initials

    #Strip html tags
    def strip_html_tags(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    #Removing accented characters
    def remove_accented_chars(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    #Removing emails
    def remove_emails(self, text: str) -> str:
        regex = r"\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+"
        # text =  text.apply(lambda x: re.sub(regex,' ', str(x)))
        text =  re.sub(regex,' ', text)
        return text

    #Removing specific pattern
    def remove_specific_pattern(self, text: str) -> str:
        text =  re.sub(self.regex_terms_to_remove,' ', text)
        return text

    #Removing usernames
    def remove_usernames(self, text: str) -> str:
        text = re.sub(self.regex_usernames_to_remove,' ', text)
        return text

    #Expanding Contractions - Contractions are shortened version of words or syllables
    def expand_contractions(self, text: str, contraction_mapping=contractions_dict) -> str:
        contractions_pattern = re.compile(
                '({})'.format('|'.join(contraction_mapping.keys())),
                flags=re.IGNORECASE|re.DOTALL
            )
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())
            try:
                expanded_contraction = first_char + expanded_contraction[1:]
            except TypeError:
                expanded_contraction = match
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    #Removing Custom words
    def remove_custom_words(self, text: str, remove_digits=False) -> str:
        """remove links, urls, emails"""
        pattern = r"[^\s]*\.(com|org|net)\S*" if not remove_digits else r"[^\s]*\.(com|org|net)\S*"
        text = re.sub(pattern, '', text)
        return text

    #Removing Special Characters
    def remove_special_characters(self, text: str, remove_digits=False) -> str:
        pattern = r"[^A-Za-z0-9\s]+" if not remove_digits else r"[^A-Za-z\s]+"
        text = re.sub(pattern, '', text)
        return text

    #Stemming
    def simple_stemmer(self, text: str) -> str:
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    #Lemmatization
    def lemmatize_text(self, text: str) -> str:
        text = self.nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    #Removing Stopwords
    def remove_stopwords(self, text: str, is_lower_case=False) -> str:
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in self.stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def combine_target_variables(self, row, count=3, original_target_col="Assignment group", count_col="tmp_target_count"):
        if row[count_col] <= count:
            return "OTHER"
        else:
            return row[original_target_col]

    def clean_corpus(
            self,
            doc: str,
            html_stripping=True,
            contraction_expansion=True,
            accented_char_removal=True,
            text_lower_case=True,
            custom_word_removal=True,
            text_lemmatization=True,
            special_char_removal=True,
            stopword_removal=True,
            remove_digits=True
        ) -> str:
        doc_log = " ".join(f"{doc}".split())
        # print(f"processing: {type(doc)}, {doc_log}")

        if not isinstance(doc, str):
            return ""

        # lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # for doc in corpus:
        doc = self.remove_specific_pattern(doc)
        doc = self.remove_usernames(doc)

        # remove stopwords
        if stopword_removal:
            doc = self.remove_stopwords(doc, is_lower_case=text_lower_case)

        # strip HTML
        if html_stripping:
            doc = self.strip_html_tags(doc)

        # remove accented characters
        if accented_char_removal:
            doc = self.remove_accented_chars(doc)

        # remove links, urls, emails
        if custom_word_removal:
            doc = self.remove_custom_words(doc)

        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!_}/])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = self.remove_special_characters(doc, remove_digits=remove_digits)

        # expand contractions
        if contraction_expansion:
            doc = self.expand_contractions(doc)


        # lemmatize text
        if text_lemmatization:
            doc = self.lemmatize_text(doc)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        doc = doc.strip()

        # print(f"cleaned doc: {type(doc)} -- {len(doc)} -- {doc}\n")
        return doc
