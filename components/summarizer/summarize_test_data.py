import summarizer_utils as sutils
import pickle
import nltk
from nltk.tokenize.moses import MosesDetokenizer

DATA_PATH = '../../data/'

models = ['no_coverage', 'more_coverage']

articles, ids = sutils.load_test_data(DATA_PATH + 'test_data.csv')

summaries = {}

for M in models:
	summarizer_internal_pickle = DATA_PATH + "pickles/decoded_stories_" + M + ".pickle"
	sutils.run_summarization_model_decoder(summarizer_internal_pickle, 
           data_path = DATA_PATH + "converted_articles/chunked/test_*" ,
           vocab_path = DATA_PATH + "summarizer_training_data/finished_files/vocab",
           log_root = DATA_PATH + "summarizer_models",
           exp_name = M)

	# Load generated summaries:
	summarization_output = pickle.load(open(summarizer_internal_pickle, "rb" ))
	tokenized_summaries = sutils.try_fix_upper_case_for_summaries(articles, summarization_output['summaries_tokens'])

	detokenizer = MosesDetokenizer()

	detokenized_summaries = []

	for s in tokenized_summaries:
	    s_detok = detokenizer.detokenize(s, return_str=True)
	    detokenized_summaries.append(s_detok)

	# Fix missing data in the original test set:
	n = 1991
 	detokenized_summaries.insert(n, '')

	summaries[M] = detokenized_summaries


summaries_3sent = sutils.get_3_sentence_summaries(articles)

summarizer_output_pickle = "../../data/pickles/summarizer_output.pickle"

summarizer_output = {
    'ids' : ids,
    'stories': articles,
    'summaries_no_coverage': summaries['no_coverage'],
    'summaries_more_coverage': summaries['more_coverage'],
    'summaries_3sent': summaries_3sent
}

pickle.dump(summarizer_output, open(summarizer_output_pickle, "wb"))