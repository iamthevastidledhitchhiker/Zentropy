# https://github.com/codelucas/newspaper
# pip3 install newspaper3k

# Fetch 30 articles:
import summarizer_utils as sutils
import story_converter as sconv
import pickle

urls = [
    "https://yle.fi/uutiset/osasto/news/finnish_capital_planning_rules_for_airbnb_market/10139225",
    "http://metropolitan.fi/entry/finnish-amer-sports-acquires-peak-performance-athletic-apparel-brand",
    "http://metropolitan.fi/entry/koskenkorva-factory-finland-12-hour-shift",
    "http://metropolitan.fi/entry/nokia-posts-abysmal-result-for-q1-2018-expects-improvements-in-q3",
    "https://yle.fi/uutiset/osasto/news/finnair_posts_rare_first-quarter_profits_plans_15_capacity_boost/10177010",
    "http://www.helsinkitimes.fi/business/15498-stockmann-to-shutter-over-20-loss-making-lindex-stores.html",
    "http://www.helsinkitimes.fi/business/15491-nokia-full-of-optimism-despite-lacklustre-first-quarter.html",
    "http://www.helsinkitimes.fi/business/15464-fortum-to-install-the-largest-rooftop-solar-power-system-in-nordics.html",
    "http://www.helsinkitimes.fi/business/15453-yit-to-lay-off-up-to-120-employees-in-finland.html",
    "http://www.helsinkitimes.fi/business/15397-nordea-confirms-re-location-to-finland.html",
    "http://www.helsinkitimes.fi/business/15362-nokia-to-help-build-first-4g-network-on-the-moon.html",
    "http://www.dailyfinland.fi/business/5175/Volkswagen-records-strong-Q1-despite-diesel-scandal",
    "http://www.dailyfinland.fi/business/5165/Mein-Schiff-1-delivered-to-German-TUI-Cruises",
    "http://www.dailyfinland.fi/business/5162/UPM-gains-first-RSB-low-ILUC-risk-certification",
    "http://www.dailyfinland.fi/business/5150/Finnish-economy-to-stagnate-after-two-years-Nordea",
    "http://www.dailyfinland.fi/business/5204/Nokia-reports-decline-in-profits-pledges-to-improve",
    "https://www.theguardian.com/technology/2018/may/01/apple-second-quarterly-report-best-ever-iphone-x",
    "http://www.dailyfinland.fi/business/5014/New-Volkswagen-CEO-vows-to-speed-up-corporate-reforms",
    "http://www.dailyfinland.fi/business/5006/airBaltic-posts-best-ever-operational-results-in-2017",
    "http://www.dailyfinland.fi/business/4999/IAG-buys-4.6-stake-in-Norwegian-airlines",
    "http://www.bbc.com/news/business-43943848",
    "http://www.bbc.com/news/business-43945254",
    "http://www.bbc.com/news/technology-43404071",
    "http://www.bbc.com/news/business-43954352",
    "http://www.bbc.com/news/business-43956968",
    "http://www.bbc.com/news/business-43959468",
    "http://www.bbc.com/news/business-43841922",
    "http://www.bbc.com/news/business-43885516",
    "http://www.bbc.com/news/business-43954262",
    "http://www.bbc.com/news/business-43960998"
]

print("Downloading articles...")
story_data = sutils.fetch_stories(urls)
print("Downloading articles DONE")

summarizer_internal_pickle = "pickles/decoded_stories.pickle"
sconv.process_and_save_to_disk(story_data['stories'], "test.bin")
sutils.run_summarization_model_decoder(summarizer_internal_pickle)

summarization_output = pickle.load(open(summarizer_internal_pickle, "rb" ))
print(summarization_output['summaries'])

tokenized_summaries = sutils.try_fix_upper_case_for_summaries(story_data['stories'], summarization_output['summaries_tokens'])

summarizer_output_pickle = "pickles/summarizer_output.pickle"

summarizer_output = {
    'urls' : story_data['urls'],
    'stories' : story_data['stories'],
    'summaries_extractive' : story_data['summaries_extractive'],
    'summaries_model' : tokenized_summaries
}

pickle.dump(summarizer_output, open(summarizer_output_pickle, "wb"))