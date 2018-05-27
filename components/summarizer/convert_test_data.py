import story_converter as sconv
import pandas as pd
import re

DATA_PATH = '../../data/'

def remove_date_title_and_id(s):
    _s = s.split('\n')
    return ''.join(_s[3:])


# replace period followed by a capital with a period space capital
def fix_periods(s):
    fixed = re.sub('\.([A-Z])', '. \g<1>', s)
    return fixed


def load_test_data(csv_file):
    article_df = pd.read_csv(csv_file)

    articles = article_df['full_text'].apply(remove_date_title_and_id)
    articles = articles.apply(fix_periods)

    ids = article_df['docno_x']

    return articles, ids


# Load test set from data folder:
articles, ids = load_test_data(DATA_PATH + 'test_data.csv')

# Convert test set articles to binary format needed for the neural summarizer:
sconv.process_and_save_to_disk(articles, "test.bin", f"{DATA_PATH}/converted_articles", verbose = True)