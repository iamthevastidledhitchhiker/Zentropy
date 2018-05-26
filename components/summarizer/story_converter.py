import os
import struct
import subprocess
from tensorflow.core.example import example_pb2

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


# Takes article as an input and stores it as a file readable by the framework
def process_and_save_to_disk(stories, out_file, finished_files_dir, verbose = False):
    stories = tokenize_stories(stories, verbose)
    stories = format_stories(stories)
    tf_example_strs = stories2examples(stories)
    save2bin(tf_example_strs, out_file, finished_files_dir)
    chunk_file(out_file, finished_files_dir)


# Story is a string
# return story as a string or story written to file?
def tokenize_stories(stories, verbose = False):
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines']
    stories_tokenized = []
    for n, story in enumerate(stories):
        (out, err) = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate(
            story.encode('utf-8'))
        stories_tokenized.append(out.decode())
        if verbose and not (n % 25):
            print(f"Processing story number: {n+1}")
    return stories_tokenized


def stories2examples(stories):
    tf_example_strs = []
    for story in stories:
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([story.encode()])
        # encode empty string as abstract
        tf_example.features.feature['abstract'].bytes_list.value.extend(["".encode()])
        tf_example_str = tf_example.SerializeToString()
        tf_example_strs.append(tf_example_str)
    return tf_example_strs


def save2bin(tf_example_strs, out_file, finished_files_dir):
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    out_file = os.path.join(finished_files_dir, out_file)
    with open(out_file, 'wb') as writer:
        for tf_example_str in tf_example_strs:
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))  # convert object to serialized c structs
            writer.write(struct.pack('%ds' % str_len, tf_example_str))


def format_stories(stories):
    stories_formatted = []
    for story in stories:
        lines = story.splitlines()

        # Lowercase everything
        lines = [line.lower() for line in lines]

        # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image
        # captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
        lines = [fix_missing_period(line) for line in lines]

        # Make article into a single string
        story = ' '.join(lines)

        stories_formatted.append(story)

    return stories_formatted


def fix_missing_period(line):
    """Adds a period to a line that is missing a period
    """
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


# Chunks the data file with *bin extension into chunks
def chunk_file(file_name, finished_files_dir):
    # Make a dir to hold the chunks
    chunks_dir = os.path.join(finished_files_dir, "chunked")
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    in_file = '%s/%s' % (finished_files_dir, file_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    set_name = os.path.splitext(file_name)[0] # get the file name without the extension
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1
