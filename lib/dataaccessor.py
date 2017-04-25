import string, nltk, csv, json
from lib.mediawikiapi import MediaWikiCommonsAPI

# enumeration of punctuation
translator = {ord(c): None for c in string.punctuation}

def get_img_mediawiki_uid(commons_url):
    return commons_url[commons_url.rfind('/')+1:]


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
            img_description = MediaWikiCommonsAPI.getImageDescription(img_uid).lower()
            img_description = img_description.translate(translator)

            img_description_text = img_description + " " + img_caption
            imgs_hash[img_uid] = i
            imgs_content.append(img_description_text)
            i += 1


    return imgs_hash, imgs_content


def write_2_csv(list, csvfile_name):
    with open(csvfile_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for score in list:
            csvwriter.writerow(score)

def get_json_article_file(int_id):
    article_file_name = "articles/{}.json".format(int_id)
    article_file = open(article_file_name, 'r')
    return json.load(article_file)


def get_jsonObj_article_content(int_id):
    article_textfile_name = "articles/{}-text.json".format(int_id)
    article_textfile = open(article_textfile_name, 'r')
    return json.load(article_textfile)