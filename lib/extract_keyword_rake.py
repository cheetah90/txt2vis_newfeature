import rake

from lib.utils import tokenize

# Using Rake for keyword extraction
def extract_keywords_from_news_content(news_content):
    # first attempt to extra keywords
    rake_object = rake.Rake("SmartStoplist.txt", 3, 2, 2)
    keywords = rake_object.run(news_content)

    if len(keywords)<=10: # if keywords are too few
        if len(news_content.split(' ')) < 110:
            rake_object = rake.Rake("SmartStoplist.txt", 3, 2, 1)
        else:
            rake_object = rake.Rake("SmartStoplist.txt", 2, 2, 2)

        keywords = rake_object.run(news_content)

    rake_keywords = []
    for prase in keywords[:10]:
        for word in prase[0].split(' '):
            rake_keywords.append(word)
    #rake_keywords = [word[0] for word in keywords]

    return rake_keywords

    # # check if the keywords are in topical sentences
    # topic_sentence = news_content.split('\n', 1)[0]
    # stemmed_tokens = tokenize(topic_sentence)
    # final_keywords = []
    # for keyword_prase in rake_keywords:
    #     keywords = keyword_prase.split(" ")
    #     for keyword in keywords:
    #         if keyword in stemmed_tokens:
    #             final_keywords.append(keyword)
    #
    # return final_keywords