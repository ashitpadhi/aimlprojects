from textblob import TextBlob
import pycountry
from langdetect import detect
import os
import re
import sys, time

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
# %matplotlib inline

#Importing libraries for text pre-processing
import spacy
from spacy_langdetect import LanguageDetector
import nltk
nltk.download('stopwords')
from nltk.tokenize.toktok import ToktokTokenizer
#from nltk.tokenize import word_tokenize
from contractions import contractions_dict
import unicodedata
from typing import Dict, List

from src.settings import settings
from src.helpers.preprocessor_helpers import PreprocessorHelper


def clean_dataset(**clean_dataset_options):
    print("clean_dataset kwargs:", clean_dataset_options)
    df = pd.read_excel(f"{settings['data_path']}/{clean_dataset_options['input_file']}")

    # get subset of data for testing the preprocessor pipeline
    # df = df.copy().iloc[1:10]

    stopword_usernames = df.Caller.to_list()
    stopword_usernames_separate = sum( [ii.split() for ii in stopword_usernames] ,  [])

    stopword_usernames = df.Caller.to_list()
    specific_patterns_to_remove = [
        r"received from:",
        r"name:.*\n",
        r"language:",
        r"browser:",
        r"email:",
        r"customer number:",
        r"summary:",
        r"click here",
        r"SID_[0-9]*",
        r"ticket_no_[0-9]*",
    ]
    # bounding condition to regex_terms_to_remove too
    kwargs = {
        "regex_terms_to_remove": r"|".join(specific_patterns_to_remove),
        "regex_usernames_to_remove": r"\b(?:" + r"|".join(stopword_usernames_separate) + r")\b",
        "specific_stopwords": ['hello', 'hi', 'please', "best", "kind", "etc"],
    }
    preprocessor_helper = PreprocessorHelper(**kwargs)

    initials = []
    for name in stopword_usernames:
        initials.append(preprocessor_helper.get_name_initials(name))

    # adding user name initials to stopwords
    preprocessor_helper.extend_stopword_list(initials)
    preprocessor_helper.extend_stopword_list(["yesnona"])

    # df['description_cleaned'] = df["Description"].apply(preprocessor_helper.clean_corpus)
    # df['short_description_cleaned'] = df["Short description"].apply(preprocessor_helper.clean_corpus)
    df["tmp_target_count"] = df.groupby(["Assignment group"])["Description"].transform("count")

    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    # df['spacy_col'] = df['description_cleaned'].apply(lambda x: nlp(x))

    total_rows = len(df.index)
    for index, row in df.iterrows():
        # printing progress
        sys.stdout.write(f"\r# processing {index} of {total_rows}  ")
        sys.stdout.flush()

        cleaned_description = preprocessor_helper.clean_corpus(row["Description"])
        cleaned_short_description = preprocessor_helper.clean_corpus(row["Short description"])
        df.loc[index, 'description_cleaned'] = cleaned_description
        df.loc[index, 'short_description_cleaned'] = cleaned_short_description
        df.loc[index, "target1"] = preprocessor_helper.combine_target_variables(
            row,
            count=30,
            original_target_col="Assignment group",
            count_col="tmp_target_count")

        # detecting language of description
        spacy_obj = (lambda x: nlp(x))(cleaned_description)
        row_lang = spacy_obj._.language
        try:
            TextBlob_lang = TextBlob(detect(cleaned_description))
        except:
            TextBlob_lang = "error"

        df.loc[index, "language"] = row_lang.get('language')
        df.loc[index, "score"] = row_lang.get('score')
        df.loc[index, "lang_textblob"] = str(TextBlob_lang)

    print("\nwriting processed records to output files...")

    df_en = df[df['lang_textblob']=='en']
    df_en.reset_index(inplace=True)
    total_rows = len(df_en.index)
    for index, row in df_en.iterrows():
        # printing progress
        sys.stdout.write(f"\r# grouping target columns {index} of {total_rows}  ")
        sys.stdout.flush()

        df_en.loc[index, "target1"] = preprocessor_helper.combine_target_variables(row)

    base = os.path.basename(clean_dataset_options['output_file'])
    output_file_name_ext = os.path.splitext(base)

    df_path = os.path.join(settings['data_path'], f"{output_file_name_ext[0]}-all{output_file_name_ext[1]}")
    df_en_path = os.path.join(settings['data_path'], f"{output_file_name_ext[0]}-en{output_file_name_ext[1]}")

    df.to_excel(df_path, index=False)
    df_en.to_excel(df_en_path, index=False)
    print("\ndone...")


if __name__ == "__main__":
    print("Initiating...")
    cleaned_dataset_file_name = "Input Data Synthetic CleanedV4.xlsx"
    input_dataset_file_name = "Input Data Unprocessed Orig.xlsx"
    clean_dataset(
        input_file=input_dataset_file_name,
        output_file=cleaned_dataset_file_name
    )
