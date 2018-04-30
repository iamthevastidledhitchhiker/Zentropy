from nltk import word_tokenize


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