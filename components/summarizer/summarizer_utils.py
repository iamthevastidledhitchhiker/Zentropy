from nltk import word_tokenize
from newspaper import Article
import pickle
import os.path
import sys
sys.path.append("./pointer-generator")
import run_summarization as ra
from nltk.tokenize import sent_tokenize
import hashlib
import re
import pandas as pd

def try_fix_upper_case_for_summaries(stories, summaries):
    """ Look for matching words in the source article and make sure summary tokens share the
    case when all article occurances are capitalized identically.

    This method is not 100% bulletproof, however it is fast.

    :param stories: An array of original stories
    :param summaries: An array of tokenized summaries
    :return: An array of summaries with attempted capitalization of words to match that inside the full story
    """
    upper_case_summaries = []

    for s in zip(stories, summaries):
        story = s[0]
        summary = s[1]
        story_tokenized = word_tokenize(story)

        for i, token in enumerate(summary):
            if token[0].isalpha():  # we have a token that begins with a letter
                matches = [w for w in story_tokenized if w.lower() == token]
                if matches:
                    if matches.count(matches[0]) == len(matches):
                        summary[i] = matches[0]
            else:
                continue

        upper_case_summaries.append(summary)

    return upper_case_summaries


def hasher(w):
    return hashlib.md5(w.encode()).hexdigest()[:9]


def extract_stories_and_ext_summaries(urls, force_download=False):
    titles = []
    stories = []
    summaries_extractive = []
    summaries_3sent = []

    path = 'stories/'

    for url in urls:
        # Calculate a hash for the URL:
        h = hasher(url)

        # Set file paths for each cached entry:
        title_file = path + h+".title"
        story_file = path + h+".story"
        summ_ext_file = path + h + ".summary_extractive"
        summ_3sent_file = path + h + ".summary_3sent"

        # if story exists in a file - load it from file
        if not force_download and os.path.isfile(story_file):
            with open(title_file, 'r') as file:
                article_title = file.read().replace('\n', '')
                titles.append(article_title)
            with open(story_file, 'r') as file:
                article_text = file.read().replace('\n', '')
                stories.append(article_text)
            with open(summ_ext_file, 'r') as file:
                summ_ext_text = file.read().replace('\n', '')
                summaries_extractive.append(summ_ext_text)
            with open(summ_3sent_file, 'r') as file:
                summ_3sent_text = file.read().replace('\n', '')
                summaries_3sent.append(summ_3sent_text)
        # else load it using newspaper API and store in on disk as well
        else:
            article = Article(url)
            article.download()

            # Get full article:
            article.parse()

            article.nlp()

            # Do some processing:
            article_text = re.sub(r"([a-z])\n\n([A-Z])", r"\1.\n\n\2", article.text)

            # Split article into sentences and
            sent_tokens = sent_tokenize(article_text)

            # Some articles begin with image caption sentence or video playback info.
            # In such case we want to remove it:

            for idx, t in enumerate(sent_tokens):
                if re.match(r"^Image(:|\s).*", t):
                    sent_tokens.pop(idx)
                if re.match(r"^Media playback(:|\s).*", t):
                    sent_tokens.pop(idx)

            article_text = ' '.join(sent_tokens)
            summ_3sent_text = ' '.join(sent_tokens[:3])

            titles.append(article.title)
            stories.append(article_text)
            summaries_extractive.append(article.summary)
            summaries_3sent.append(summ_3sent_text)

            with open(title_file, "w") as file:
                file.write(article.title)

            with open(story_file, "w") as file:
                file.write(article.text)

            with open(summ_ext_file, "w") as file:
                file.write(article.summary)

            with open(summ_3sent_file, "w") as file:
                file.write(summ_3sent_text)

    return titles, stories, summaries_extractive, summaries_3sent


def fetch_and_pickle_stories(urls, force_download=False):
    """Fetches news stories from URLs provided as an array"""
    RAW_STORIES_PICKLE_FILE = 'pickles/raw_stories.pickle'

    titles, stories, summaries_extractive, summaries_3sent = extract_stories_and_ext_summaries(urls, force_download)
    story_data = {
        'urls': urls,
        'titles': titles,
        'stories': stories,
        'summaries_extractive': summaries_extractive,
        'summaries_3sent': summaries_3sent
    }
    pickle.dump(story_data, open(RAW_STORIES_PICKLE_FILE, "wb"))
    return story_data

def load_stories_from_csv():
    pass


def run_summarization_model_decoder(pickle_file, data_path, vocab_path, log_root, exp_name):
    argv = ["entry_point",
            "--mode=decode",
            "--api_mode=1",
            "--pickle_file=" + pickle_file,
            "--single_pass=1",
            "--data_path=" + data_path,
            "--vocab_path=" + vocab_path,
            "--log_root=" + log_root,
            "--exp_name=" + exp_name]

    try:
        print("Starting TensorFlow Decoder...")
        ra.run_external(argv)
    except SystemExit:
        print("Summarization model exited as expected :)")


def get_3_sentence_summaries(articles):
    summaries = []
    for article in articles:
        sent_tokens = sent_tokenize(article)
        summaries.append(' '.join(sent_tokens[:3]))
    return summaries

# Test data cleaning functionality:
def remove_date_title_and_id(s):
    _s = s.split('\n')
    return ''.join(_s[3:])


#replace period followed by a capital with a period space capital
def fix_periods(s):
    fixed = re.sub('\.([A-Z])', '. \g<1>', s)
    return fixed

def load_test_data(csv_file):
    article_df = pd.read_csv(csv_file)
    
    articles = article_df['full_text'].apply(remove_date_title_and_id)
    articles = articles.apply(fix_periods)
    
    ids = article_df['docno_x']
    
    return articles, ids
