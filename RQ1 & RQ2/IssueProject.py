import os
import re
import csv
import time
from datetime import datetime
from collections import Counter

import pandas as pd
import requests
import textstat
import nltk
import langdetect
import spacy
import textdescriptives as td

from langdetect import detect, DetectorFactory
from nltk.corpus import wordnet as wn
from textblob import TextBlob


nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textdescriptives/all")

start_time = time.time()

nltk.download("wordnet")
nltk.download("omw-1.4")

DetectorFactory.seed = 0


def is_english(text):
    """
    Check whether a text is written in English.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    bool
        True if the detected language is English, otherwise False.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    try:
        return detect(text) == "en"
    except langdetect.lang_detect_exception.LangDetectException:
        return False


def get_issue_events(api_url):
    """
    Retrieve the event history of a GitHub issue through the GitHub API.

    Parameters
    ----------
    api_url : str
        GitHub API URL for the issue events endpoint.

    Returns
    -------
    list or None
        JSON response containing issue events if the request succeeds,
        otherwise None.
    """
    headers = {
        "Authorization": f"token {token_github}"
    }

    response = requests.get(api_url, headers=headers)

    if response.history:
        print(f"Request was redirected. History: {response.history}")

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:
        print(f"Request limit reached: {response.status_code}")
        print("Sleeping before retrying...")
        time.sleep(2400)
    else:
        print(f"Request error: {response.status_code}")
        return None


def window_time(unlabeled_time, labeled_time_for_unlabeled_label):
    """
    Compute the time difference between label removal and the previous addition
    of the same label.

    Parameters
    ----------
    unlabeled_time : str
        Timestamp when the label was removed.
    labeled_time_for_unlabeled_label : str
        Timestamp when the same label was previously added.

    Returns
    -------
    datetime.timedelta or None
        Time difference between the two events if both timestamps are available,
        otherwise None.
    """
    if unlabeled_time and labeled_time_for_unlabeled_label:
        unlabeled_time_dt = datetime.strptime(unlabeled_time, "%Y-%m-%dT%H:%M:%SZ")
        labeled_time_for_unlabeled_label_dt = datetime.strptime(
            labeled_time_for_unlabeled_label,
            "%Y-%m-%dT%H:%M:%SZ"
        )

        return unlabeled_time_dt - labeled_time_for_unlabeled_label_dt

    return None


def get_label_added_datetime(events, label_name):
    """
    Find the timestamp when a specific label was added to an issue.

    Parameters
    ----------
    events : list
        List of GitHub issue events.
    label_name : str
        Name of the label to search for.

    Returns
    -------
    str or None
        Timestamp of the label addition if found, otherwise None.
    """
    for event in events:
        if event["event"] == "labeled" and event["label"]["name"] == label_name:
            return event["created_at"]

    return None


def is_ambiguous(word):
    """
    Check whether a word is ambiguous according to WordNet.

    Parameters
    ----------
    word : str
        Word to analyze.

    Returns
    -------
    bool
        True if the word has more than one WordNet synset, otherwise False.
    """
    return len(wn.synsets(word)) > 1


def calculate_ambiguity_index(text):
    """
    Calculate the ambiguity index of a text.

    The ambiguity index is the proportion of words that have more than one
    WordNet synset.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    float
        Proportion of ambiguous words in the text.
    """
    words = text.split()
    ambiguous_words = [word for word in words if is_ambiguous(word)]
    total_ambiguous_words = len(ambiguous_words)
    total_words = len(words)

    return total_ambiguous_words / total_words if total_words > 0 else 0


informal_words = [
    "lol", "btw", "omg", "u", "ur", "lmao", "brb", "idk", "smh",
    "tbh", "gr8", "b4", "wanna", "gonna", "gotta", "kinda", "sorta",
    "ain't", "y'all", "wassup", "yo", "dunno", "lemme", "coz",
    "haha", "hehe", "omw", "imo", "imho", "bro", "dude", "homie", "bae",
    "lit", "fam", "chill", "cool", "dope", "savage", "fyi", "af", "ttyl",
    "np", "fomo", "yolo", "dm", "imma", "wow", "yay", "nah", "uh-huh",
    "oh", "huh", "ouch", "ew", "phew", "coz", "lemme", "gonna", "hafta",
    "wow", "nah", "bro", "dude", "fam", "dope", "lit", "savage", "omg",
    "fomo", "don't", "can't", "won't", "shouldn't", "couldn't", "wouldn't",
    "isn't", "aren't", "wasn't", "weren't", "I'm", "you're", "he's",
    "she's", "it's", "we're", "they're", "I've", "you've", "we've", "they've",
    "I'd", "you'd", "he'd", "she'd", "we'd", "they'd", "my", "your",
    "his", "her", "its", "our", "their", "yeah", "nah", "yup", "nope",
    "alright", "ok", "okay", "anyways", "like", "buddy", "mate", "gal",
    "guy", "kiddos", "cool", "awesome", "great", "amazing", "fantastic",
    "chill", "rad", "dope", "legit", "sweet", "neat", "fab", "chillax", "fella",
    "chillin'", "flex", "hangout", "nerd", "geek", "jock", "peeps", "squad",
    "homie", "kickback", "low-key", "high-key", "selfie", "hashtag", "viral", "meme",
    "troll", "lurk", "tweet", "post", "bio", "unfollow", "subscribe", "vlog", "blog",
    "story", "hey", "hi", "howdy", "yo", "sup", "YOLO", "TMI", "SMH", "BFF", "BRB",
    "ASAP", "L8R", "TYT", "GTFO", "TTYL", "TLDR", "TTYT", "OMW",
    "NSFW", "ICYMI", "FWIW", "FWB"
]


def count_informal_words(text):
    """
    Count informal words in a text.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    int
        Number of informal words found in the text.
    """
    words = text.split()
    informal_count = sum(1 for word in words if word.lower() in informal_words)

    return informal_count


def calculate_informality_index(text):
    """
    Calculate the informality index of a text.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    float
        Proportion of informal words in the text.
    """
    words = text.split()
    total_words = len(words)
    informal_count = count_informal_words(text)

    return informal_count / total_words if total_words > 0 else 0


def count_paragraphs(text):
    """
    Count non-empty paragraphs in a text.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    int
        Number of non-empty paragraphs.
    """
    paragraphs = text.split("\n")

    return len([paragraph for paragraph in paragraphs if paragraph.strip()])


def calculate_flesch_reading_ease(text):
    """
    Calculate the Flesch Reading Ease score.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    float
        Flesch Reading Ease score clipped between 0 and 100.
    """
    if not text.strip():
        return 0

    score = textstat.flesch_reading_ease(text)

    if score > 100:
        return 100
    elif score < 0:
        return 0

    return score


def calculate_flesch_kincaid_grade(text):
    """
    Calculate the Flesch-Kincaid Grade Level score.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    float
        Flesch-Kincaid Grade Level score clipped between 0 and 100.
    """
    if not text.strip():
        return 0

    score = textstat.flesch_kincaid_grade(text)

    if score > 100:
        return 100
    elif score < 0:
        return 0

    return score


vague_words = [
    "something", "anything", "everything", "nothing", "thing", "stuff",
    "many", "much", "few", "several", "a lot", "a few", "some",
    "a number of", "various", "probably", "possibly", "maybe",
    "perhaps", "likely", "uncertain", "could be", "might",
    "someone", "anyone", "no one", "somebody", "anybody", "everybody",
    "nobody", "someone else", "good", "bad", "nice", "poor", "better",
    "worse", "best", "worst", "less", "more", "often", "rarely",
    "sometime", "occasionally", "frequently", "sometimes",
    "for example", "such as", "like", "in general", "as a rule",
    "to some extent", "more or less", "around", "approximately",
    "about", "nearly", "close to", "roughly", "this", "that",
    "these", "those", "it", "they", "them", "I think", "I believe",
    "I suppose", "I guess", "to my knowledge", "it seems", "in my opinion",
    "people", "individuals", "experts", "authorities", "users",
    "clients", "customers", "etc.", "and so on", "and others", "or something"
]


def calculate_vagueness_index(text):
    """
    Calculate the vagueness index of a text.

    The vagueness index is computed as the proportion of vague words found in
    the text.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    float
        Proportion of vague words in the text.
    """
    text = re.sub(r"\W+", " ", text.lower())
    words = text.split()

    if not words:
        return 0

    word_count = Counter(words)
    vague_word_count = sum(word_count[word] for word in vague_words if word in word_count)

    return vague_word_count / len(words)


def get_issue_url(api_url):
    """
    Convert a GitHub API repository URL into a standard GitHub URL.

    Parameters
    ----------
    api_url : str
        GitHub API URL.

    Returns
    -------
    str
        Standard GitHub URL.
    """
    original_url = api_url.replace("api.github.com/repos", "github.com")

    return original_url


correctly_classified_csv = "correctly_class.csv"
miss_classified_csv = "miss_classified.csv"

token_github = os.getenv("GITHUB_TOKEN")

if not token_github:
    raise ValueError("Missing GitHub token. Set it as an environment variable named GITHUB_TOKEN.")

df = pd.read_csv("dataset.csv")

subset_df = df.iloc[722898:]

for index, row in subset_df.iterrows():
    title = df.at[index, "issue_title"]
    body = df.at[index, "issue_body"]

    if pd.isna(body):
        body = ""

    if pd.isna(title):
        title = ""

    if body:
        if not is_english(body):
            continue
    else:
        if not is_english(title):
            continue

    issue_api_url = df.at[index, "issue_url"]
    next_index = index + 1

    events_url = issue_api_url + "/events"
    events = get_issue_events(events_url)

    if events:
        title_word_length = len(str(title).split())
        title_character_length = len(str(title).replace(" ", ""))
        title_blob = TextBlob(title)
        title_polarity = title_blob.sentiment.polarity
        title_subjectivity = title_blob.sentiment.subjectivity
        title_ambiguity_index = calculate_ambiguity_index(title)
        title_informality_index = calculate_informality_index(title)
        title_flesch_reading_ease = calculate_flesch_reading_ease(title)
        title_flesch_kincaid_grade = calculate_flesch_kincaid_grade(title)

        body_word_length = len(str(body).split())
        body_character_length = len(str(body).replace(" ", ""))
        body_blob = TextBlob(body)
        body_polarity = body_blob.sentiment.polarity
        body_subjectivity = body_blob.sentiment.subjectivity
        num_paragraphs = count_paragraphs(body)
        body_ambiguity_index = calculate_ambiguity_index(body)
        body_informality_index = calculate_informality_index(body)
        body_flesch_reading_ease = calculate_flesch_reading_ease(body)
        body_flesch_kincaid_grade = calculate_flesch_kincaid_grade(body)
        vagueness_index = calculate_vagueness_index(body)

        doc = nlp(body)
        bullet_points = td.components.quality.proportion_bullet_points(doc)
        alpha_ratio = td.components.quality.alpha_ratio(doc)
        mean_word_length = td.components.quality.mean_word_length(doc)
        stop_words = td.components.quality.n_stop_words(doc)
        ellipsis_proportion = td.components.quality.proportion_ellipsis(doc)
        coherence = doc._.coherence
        first_order_coherence = coherence["first_order_coherence"]
        entropy = doc._.entropy
        perplexity = doc._.perplexity
        pos = doc._.pos_proportions
        pos_prop_ADJ = pos["pos_prop_ADJ"]
        pos_prop_ADP = pos["pos_prop_ADP"]
        pos_prop_ADV = pos["pos_prop_ADV"]
        pos_prop_AUX = pos["pos_prop_AUX"]
        pos_prop_NOUN = pos["pos_prop_NOUN"]
        pos_prop_NUM = pos["pos_prop_NUM"]
        pos_prop_VERB = pos["pos_prop_VERB"]
        pos_prop_PRON = pos["pos_prop_PRON"]
        descriptive_stats = doc._.descriptive_stats
        lexical_diversity_less = descriptive_stats["proportion_unique_tokens"]

        print(issue_api_url)

        events_df = pd.DataFrame(events)
        labeled_events = events_df[events_df["event"] == "labeled"]
        unlabeled_events = events_df[events_df["event"] == "unlabeled"]

        if unlabeled_events.empty:
            with open(correctly_classified_csv, mode="a", newline="", encoding="utf-8") as file_csv:
                writer = csv.writer(file_csv)
                writer.writerow([
                    issue_api_url,
                    title_word_length,
                    title_character_length,
                    title_polarity,
                    title_subjectivity,
                    title_ambiguity_index,
                    title_informality_index,
                    title_flesch_reading_ease,
                    title_flesch_kincaid_grade,
                    body_word_length,
                    body_character_length,
                    body_polarity,
                    body_subjectivity,
                    num_paragraphs,
                    body_ambiguity_index,
                    body_informality_index,
                    body_flesch_reading_ease,
                    body_flesch_kincaid_grade,
                    vagueness_index,
                    bullet_points,
                    alpha_ratio,
                    mean_word_length,
                    stop_words,
                    ellipsis_proportion,
                    first_order_coherence,
                    entropy,
                    perplexity,
                    pos_prop_ADJ,
                    pos_prop_ADP,
                    pos_prop_ADV,
                    pos_prop_AUX,
                    pos_prop_NOUN,
                    pos_prop_NUM,
                    pos_prop_VERB,
                    pos_prop_PRON,
                    lexical_diversity_less
                ])

        for _, row in unlabeled_events.iterrows():
            unlabeled_label_name = row["label"]["name"]
            unlabeled_label_time = row["created_at"]
            flag = False

            for _, row in labeled_events.iterrows():
                labeled_label_name = row["label"]["name"]
                labeled_label_time = row["created_at"]

                if labeled_label_name != unlabeled_label_name and unlabeled_label_time <= labeled_label_time:
                    labeled_time_for_unlabeled_label = get_label_added_datetime(events, unlabeled_label_name)
                    window = window_time(unlabeled_label_time, labeled_time_for_unlabeled_label)

                    if window is None or window.seconds <= 60:
                        continue

                    with open(miss_classified_csv, mode="a", newline="", encoding="utf-8") as file_csv:
                        writer = csv.writer(file_csv)
                        writer.writerow([
                            issue_api_url,
                            unlabeled_label_name,
                            window,
                            title_word_length,
                            title_character_length,
                            title_polarity,
                            title_subjectivity,
                            title_ambiguity_index,
                            title_informality_index,
                            title_flesch_reading_ease,
                            title_flesch_kincaid_grade,
                            body_word_length,
                            body_character_length,
                            body_polarity,
                            body_subjectivity,
                            num_paragraphs,
                            body_ambiguity_index,
                            body_informality_index,
                            body_flesch_reading_ease,
                            body_flesch_kincaid_grade,
                            vagueness_index,
                            bullet_points,
                            alpha_ratio,
                            mean_word_length,
                            stop_words,
                            ellipsis_proportion,
                            first_order_coherence,
                            entropy,
                            perplexity,
                            pos_prop_ADJ,
                            pos_prop_ADP,
                            pos_prop_ADV,
                            pos_prop_AUX,
                            pos_prop_NOUN,
                            pos_prop_NUM,
                            pos_prop_VERB,
                            pos_prop_PRON,
                            lexical_diversity_less
                        ])

                    flag = True
                    break

            if flag:
                break

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
