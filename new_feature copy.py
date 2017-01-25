import pandas, json, requests, nltk, string, math, csv
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk.data

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
            except KeyError:
                Categories = ''

            try:
                ImgDescription = response['query']['pages'][pageid]['imageinfo'][0]['extmetadata']['ImageDescription']['value']
            except KeyError:
                ImgDescription = ''

            description = Objname + " " + Categories + " " + ImgDescription

        return description


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


# def from_caption_keywords(caption_token, news_tokens):
#     # tagging part of speech
#     partofspeech = nltk.pos_tag(caption_token)
#
#     token_tfidf = []
#     max_tfidf = 0
#     max_word = ''
#     max_actual_word = ''
#
#     for each_token in partofspeech:
#         # only iterate noun for now and not a stop word
#         if each_token[0] not in stop and get_wordnet_pos(each_token[1]) != wordnet.VERB:
#
#             word_stem = nltk.stem.WordNetLemmatizer().lemmatize(each_token[0], get_wordnet_pos(each_token[1]))
#
#             try:
#                 token_synonyms = wordnik_api.getRelatedWords(word_stem, 'synonym', wordnik_apikey, useCanonical=True)
#             except requests.ConnectionError:
#                 print(ConnectionError)
#
#             # add the word itself
#             token_synonyms.append(word_stem)
#             synonym_max_tfidf = 0
#             synonym_max_word = ''
#
#             # compute max_{synonyms} (tf * idf)
#             for synonym in token_synonyms:
#                 # if synonym is not stopwords
#                 if synonym not in stop:
#                     # TODO: try boolean tf
#                     # current_tf = 1 if news_tokens.count(synonym) > 0 else 0
#                     current_tf = 1 + math.log(news_tokens.count(synonym)) if news_tokens.count(synonym) > 0 else 0
#                     current_idf = 0
#                     if current_tf != 0:
#                         current_idf = idf_api.getMinIDF(synonym)
#
#                     current_tfidf = current_tf * current_idf
#                     if current_tfidf > synonym_max_tfidf:
#                         synonym_max_tfidf = current_tfidf
#                         synonym_max_word = synonym
#
#             token_tfidf.append(synonym_max_tfidf)
#
#             if synonym_max_tfidf > max_tfidf:
#                 max_tfidf = synonym_max_tfidf
#                 max_word = word_stem
#                 max_actual_word = synonym_max_word
#
#     if max(token_tfidf) != max_tfidf:
#         assert False
#     feature_values.append(max_tfidf)
#     print("news id: {}, caption: {}, max_tf_idf: {}, word in caption: {}, actual word: {}".format(row['id'],
#                                                                                                   caption_token,
#                                                                                                   max_tfidf, max_word,
#                                                                                                   max_actual_word))

def getIDF(word):
    if word in idf_dataset:
        return idf_dataset[word]
    else:
        #print("{} does not exist in the idf dataset".format(word))
        return 0


def extract_keywords_from_news_content(news_tokens):
    # tagging the part of speech for the news articles
    partofspeech = nltk.pos_tag(news_tokens)

    lemmatized_news_text = []
    # lemmatizing the text
    for each_token in partofspeech:
        # if the token not in the stopwords
        if each_token[0] not in stop and (get_wordnet_pos(each_token[1]) == wordnet.NOUN):
            word_stem = nltk.stem.WordNetLemmatizer().lemmatize(each_token[0], get_wordnet_pos(each_token[1]))
            lemmatized_news_text.append(word_stem)
    # Use the tf approach, here the normalizer is the # of nouns, could be changed to something else later
    news_lemmatized_freqdist = nltk.FreqDist(lemmatized_news_text)
    for each_token in news_lemmatized_freqdist:
        current_idf = getIDF(each_token)
        news_lemmatized_freqdist[each_token] = current_idf * news_lemmatized_freqdist[each_token]/len(lemmatized_news_text)

    news_keywords = news_lemmatized_freqdist.most_common(10)

    return news_keywords


def from_news_keywords(caption, news_tokens):
    news_keywords = extract_keywords_from_news_content(news_tokens)


    # Method 2: use nouns in the topic sentences to
    # for each_token in partofspeech:
    #     # if the token is nouns
    #     if each_token[0] not in stop and (get_wordnet_pos(each_token[1]) == wordnet.NOUN or get_wordnet_pos(each_token[1]) == wordnet.ADJ):
    #         word_stem = nltk.stem.WordNetLemmatizer().lemmatize(each_token[0], get_wordnet_pos(each_token[1]))
    #         lemmatized_news_text.append(word_stem)
    #
    # news_lemmatized_freqdist = nltk.FreqDist(lemmatized_news_text)
    # # compute the tf-idf for each word
    # for each_token in news_lemmatized_freqdist:
    #     current_idf = getIDF(each_token)
    #     news_lemmatized_freqdist[each_token] = current_idf


    # compute scores against the image descriptions
    num_tokens = len(img_desc_tokens)
    score = 0

    for keyword in news_keywords:
        # incorporate synonyms
        # try:
        #     token_synonyms = wordnik_api.getRelatedWords(keyword[0], 'synonym', wordnik_apikey,
        #                                                              useCanonical=True, limitPerRelationshipType=5)
        # except requests.ConnectionError:
        #     print(ConnectionError)
        #
        # token_synonyms.append(keyword)

        if img_description_text.count(keyword[0]) > 0 :
            # try:
            #     current_idf = idf[keyword[0]]
            # except KeyError:
            #     current_idf = 0
            #
            score += (1 + math.log(keyword[1]) )
            print("index: {}, found keyword: {}".format(index, keyword[0]))

    print("index: {}, score: {}".format(index, score))

    return score


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
    imgs_infos = {}

    for item in jsonObj_article_file['articles']:
        if item['title'] != 'Map':
            img_caption = item['images'][0]['caption']
            img_uid = get_img_mediawiki_uid(item['images'][0]['url'])
            img_description_text = mediawiki_api.getImageDescription(img_uid).lower() + " " + img_caption
            imgs_infos[img_uid] = img_description_text

    return imgs_infos


if __name__ == "__main__":
    feature_file_df = pandas.read_csv("./outliers.csv")
    wordnik_apikey = 'f23e4fbee0bc99b46500303efeb0811cad344fed5869d370c'
    wordnik_api = WordnikAPI()
    #idf_api = IDFserverAPI()
    mediawiki_api = MediaWikiCommonsAPI()
    #load the idf dataset into the memory
    idf_dataset = load_idf_dataset()

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

        # get all the images information related to the news articles in token format:
        imgs_info = parse_imgs_info(jsonObj_article_file)


        # use the complete news content + title
        news_content = get_news_text(jsonObj_article_file)
        news_content += " "
        news_content += get_article_title(jsonObj_article_content)
        # remove punctuation
        news_content = news_content.translate(translator)
        news_tokens = nltk.word_tokenize(news_content)

        # # get the first sentence of the news article
        # sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # list_of_sentences = sentence_tokenizer.tokenize(get_news_text(jsonObj).strip())
        # topic_sentences = list_of_sentences[0] + list_of_sentences[1]
        # topic_sentences = topic_sentences.translate(translator)
        # first_sentence_tokens = nltk.word_tokenize(topic_sentences)

        # get the title of the news article
        # jsonObj = json.load(article_textfile)
        # article_title = get_article_title(jsonObj)
        # article_title = article_title.translate(translator)
        # article_title_tokens = nltk.word_tokenize(article_title)


        # from keywords in the caption
        #from_caption_keywords(caption_token, news_tokens)

        # from keywrds in the news
        score = from_news_keywords(imgs_info, news_tokens)
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

