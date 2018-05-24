import summarizer_utils as sutils

DATA_PATH = '../../data/'

models = ['no_coverage', 'some_coverage', 'more_coverage']

for M in models:
	summarizer_internal_pickle = DATA_PATH + "pickles/decoded_stories_" + M + ".pickle"
	sutils.run_summarization_model_decoder(summarizer_internal_pickle, 
           data_path = DATA_PATH + "converted_articles/chunked/test_*" ,
           vocab_path = DATA_PATH + "summarizer_training_data/finished_files/vocab",
           log_root = DATA_PATH + "summarizer_models",
           exp_name = M)