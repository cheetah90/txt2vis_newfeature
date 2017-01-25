import pandas, json, requests, nltk, string, math, csv
from collections import Counter


from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk.data
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

from bs4 import BeautifulSoup
import rake

# enumeration of punctuation
translator = {ord(c): None for c in string.punctuation}

class MediaWikiCommonsAPI:
    def __init__(self):
        self.baseUrl = 'https://commons.wikimedia.org/w/api.php?action=query&'

    def getImageDescription(self, uid):
        url = self.baseUrl + "titles={}&prop=imageinfo&iiprop=extmetadata&format=json&indexpageids".format(uid)
        response = requests.get(url)

        description = ""

        if (response.status_code == 200):
            response = response.json()
            pageid = response['query']['pageids'][0]
            try:
                Objname = response['query']['pages'][pageid]['imageinfo'][0]['extmetadata']['ObjectName']['value']
            except KeyError:
                Objname = ''

            try:
                Categories = response['query']['pages'][pageid]['imageinfo'][0]['extmetadata']['Categories']['value']
                Categories = Categories.replace("|", " ")
            except KeyError:
                Categories = ''

            try:
                ImgDescription = response['query']['pages'][pageid]['imageinfo'][0]['extmetadata']['ImageDescription']['value']
            except KeyError:
                ImgDescription = ''

            description = Objname + " " + Categories + " " + ImgDescription

        # need to clean up the image description a bit
        # remove the html elements
        clean_description = BeautifulSoup(description, "lxml").text

        # remove all numbers
        clean_description = "".join([i for i in clean_description if not i.isdigit()])

        return clean_description


def get_caption_tokens(jsonObj, record):
    for item in jsonObj['articles']:
        if item['title'] != 'Map' and item['images'][0]['url'] == record['image_url']:
            # convert to lower case for counting
            lower_case_caption = item['images'][0]['caption'].lower()
            # remove punctuation
            lower_case_caption = lower_case_caption.translate(translator)
            caption_token = nltk.word_tokenize(lower_case_caption)
            return caption_token

    return []

def get_news_text(jsonObj):
    text = jsonObj['text'].lower() #convert to lower case for matching
    return text


# from http://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class WordnikAPI:
    def __init__(self):
        self.baseUrl = 'http://api.wordnik.com:80/v4/word.json'

    def getRelatedWords(self, word, relationshipTypes, api_key, useCanonical=False, limitPerRelationshipType = 10):
        url = self.baseUrl+"/{}/relatedWords?useCanonical={}&relationshipTypes={}&limitPerRelationshipType={}&api_key={}".format(word, useCanonical, relationshipTypes,limitPerRelationshipType, api_key)
        response = requests.get(url)
        if (response.status_code == 200):
            if not response.json():
                return []

            return response.json()[0]['words']

        raise requests.ConnectionError('Did not get a 200 response.')

# def getIDF(word):
#     if word in idf_dataset:
#         return idf_dataset[word]
#     else:
#         #print("{} does not exist in the idf dataset".format(word))
#         return 0

# This is the old method that is using tf-idf type of features for keywords extraction
# def extract_keywords_from_news_content(news_content):
#     # remove punctuation
#     news_content = news_content.translate(translator)
#     news_content = nltk.word_tokenize(news_content)
#
#     # tagging the part of speech for the news articles
#     partofspeech = nltk.pos_tag(news_content)
#
#     lemmatized_news_text = []
#     # lemmatizing the text
#     for each_token in partofspeech:
#         # if the token not in the stopwords
#         if each_token[0] not in stop and (get_wordnet_pos(each_token[1]) == wordnet.NOUN):
#             word_stem = nltk.stem.WordNetLemmatizer().lemmatize(each_token[0], get_wordnet_pos(each_token[1]))
#             lemmatized_news_text.append(word_stem)
#     # Use the tf approach, here the normalizer is the # of nouns, could be changed to something else later
#     news_lemmatized_freqdist = nltk.FreqDist(lemmatized_news_text)
#     for each_token in news_lemmatized_freqdist:
#         current_idf = getIDF(each_token)
#         news_lemmatized_freqdist[each_token] = current_idf * news_lemmatized_freqdist[each_token]/len(lemmatized_news_text)
#
#     news_keywords = news_lemmatized_freqdist.most_common(10)
#
#     return news_keywords






#
# def from_news_keywords(caption, news_tokens):
#     news_keywords = extract_keywords_from_news_content(news_tokens)
#
#
#     # Method 2: use nouns in the topic sentences to
#     # for each_token in partofspeech:
#     #     # if the token is nouns
#     #     if each_token[0] not in stop and (get_wordnet_pos(each_token[1]) == wordnet.NOUN or get_wordnet_pos(each_token[1]) == wordnet.ADJ):
#     #         word_stem = nltk.stem.WordNetLemmatizer().lemmatize(each_token[0], get_wordnet_pos(each_token[1]))
#     #         lemmatized_news_text.append(word_stem)
#     #
#     # news_lemmatized_freqdist = nltk.FreqDist(lemmatized_news_text)
#     # # compute the tf-idf for each word
#     # for each_token in news_lemmatized_freqdist:
#     #     current_idf = getIDF(each_token)
#     #     news_lemmatized_freqdist[each_token] = current_idf
#
#
#     # compute scores against the image descriptions
#     num_tokens = len(img_desc_tokens)
#     score = 0
#
#     for keyword in news_keywords:
#         # incorporate synonyms
#         # try:
#         #     token_synonyms = wordnik_api.getRelatedWords(keyword[0], 'synonym', wordnik_apikey,
#         #                                                              useCanonical=True, limitPerRelationshipType=5)
#         # except requests.ConnectionError:
#         #     print(ConnectionError)
#         #
#         # token_synonyms.append(keyword)
#
#         if img_description_text.count(keyword[0]) > 0 :
#             # try:
#             #     current_idf = idf[keyword[0]]
#             # except KeyError:
#             #     current_idf = 0
#             #
#             score += (1 + math.log(keyword[1]) )
#             print("index: {}, found keyword: {}".format(index, keyword[0]))
#
#     print("index: {}, score: {}".format(index, score))
#
#     return score


def get_imgdesc_tokens(description):
    # convert to lower case for counting
    lower_case_caption = description.lower()
    # remove punctuation
    lower_case_caption = lower_case_caption.translate(translator)
    #TODO: might need to remove stop words
    text = nltk.word_tokenize(lower_case_caption)

    return text


def get_img_mediawiki_uid(commons_url):
    return commons_url[commons_url.rfind('/')+1:]


def load_idf_dataset():
    with open('./idf.txt', newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter='\t')
        idf_dataset = {rows[0].lower(): float(rows[1]) for rows in reader}

        return idf_dataset


def get_article_title(jsonObj):
    return jsonObj['title'].lower()


def parse_imgs_info(jsonObj_article_file):
    imgs_hash = {}
    imgs_content = []
    i = 0

    for item in jsonObj_article_file['articles']:
        if item['title'] != 'Map':
            img_caption = item['images'][0]['caption'].lower()
            img_caption = img_caption.translate(translator)

            img_uid = "File:"+get_img_mediawiki_uid(item['images'][0]['url'])
            img_description = mediawiki_api.getImageDescription(img_uid).lower()
            img_description = img_description.translate(translator)

            img_description_text = img_description + " " + img_caption
            imgs_hash[img_uid] = i
            imgs_content.append(img_description_text)
            i += 1


    return imgs_hash, imgs_content


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


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


def stem_imgs_content(imgs_content):
    stemmed_imgs_content = []

    for img_content_item in imgs_content:
        stemmed_tokens = tokenize(img_content_item)
        stemmed_img_content = " ".join(stemmed_tokens)
        stemmed_imgs_content.append(stemmed_img_content)

    return stemmed_imgs_content


def normalize_words(imgs_content, query_string):
    new_imgs_content = []

    for img_content in imgs_content:
        counts = Counter(img_content.split(' '))
        for keyword in query_string.split(' '):
            for key in counts.keys():
                if keyword in key:
                    img_content = img_content.replace(key, keyword)
                elif key in keyword:
                    new_query_string = query_string.replace(keyword, key)

        new_imgs_content.append(img_content)

    return (new_imgs_content, new_query_string)

def write_2_csv(list, csvfile_name):
    with open(csvfile_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for score in list:
            csvwriter.writerow(score)


if __name__ == "__main__":
    #feature_file_df = pandas.read_csv("./nonmap.csv")
    feature_file_df = pandas.read_csv("./nonmap.csv")
    mediawiki_api = MediaWikiCommonsAPI()
    #stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english")
    #idf_dataset = load_idf_dataset()

    output = [[] for i in range(0,41)]
    scores_list = []

    # get the stopwords
    stop = set(stopwords.words('english'))

    for index, row in feature_file_df.iterrows():
        # get the content of the news article
        article_file_name = "articles/{}.json".format(int(row['id']))
        article_file = open(article_file_name, 'r')
        jsonObj_article_file = json.load(article_file)

        # get the title of the news article
        article_textfile_name = "articles/{}-text.json".format(int(row['id']))
        article_textfile = open(article_textfile_name, 'r')
        jsonObj_article_content = json.load(article_textfile)

        # Extract keywords using the complete news content + title
        news_content = get_news_text(jsonObj_article_file)
        news_title = get_article_title(jsonObj_article_content) + " "
        news_content = news_title + news_content

        news_keywords = extract_keywords_from_news_content(news_content)
        query_string = " ".join(news_keywords)

        try:
            # get all the images information related to the news articles in token format:
            imgs_hash, imgs_content = parse_imgs_info(jsonObj_article_file)
        except ConnectionError as err:
            print("Connection Errors: {}".format(err))
            for score in scores_list:
                print(score)

            write_2_csv(scores_list, "before_error_results.csv")



        # add the query_string to corpus
        imgs_content = stem_imgs_content(imgs_content)
        imgs_content.append(query_string)

        # additional normalization: if keywords is a substring in image content, use the keyword as the cannonical form
        imgs_content, query_string = normalize_words(imgs_content, query_string)

        #process these images info to build tf-idf
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
        tfs = tfidf.fit_transform(imgs_content)

        # get the tfidf vector of the query string
        query_tf_idf_vector = tfidf.transform([query_string])

        # compute cosine similarity between query's tfidf and all other documents
        cosine_similarities = linear_kernel(query_tf_idf_vector, tfs).flatten()

        # get the score for this row
        uid = row['commons_url']
        uid = uid[uid.rfind('/')+1:]
        score = cosine_similarities[imgs_hash[uid]]

        print("Index: {}, Score: {}".format(index, score))

        scores_list.append(score)

        new_row_data = [row['id'], row['commons_url'], row['avg_rank'], score]

        output[int(row['id'])].append(new_row_data)

    for index in range(0, len(output)):
        with open("results_for{}.csv".format(index), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(["id", "common_url", "avg_rank", "new_feature"])
            for data in output[index]:
                csvwriter.writerow(data)

    for score in scores_list:
        print(score)


    write_2_csv(scores_list, "results_all.csv")



