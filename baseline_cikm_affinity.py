import json

import pandas
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from lib.dataaccessor import get_news_text, get_article_title, parse_imgs_info, write_2_csv, get_json_article_file, get_jsonObj_article_content
from lib.mediawikiapi import MediaWikiCommonsAPI
from lib.extract_keyword_rake import extract_keywords_from_news_content

from lib.utils import stem_imgs_content
from lib.utils import normalize_words


if __name__ == "__main__":
    feature_file_df = pandas.read_excel("./complete.xlsx")
    mediawiki_api = MediaWikiCommonsAPI()
    stemer = SnowballStemmer("english")

    output = [[] for i in range(0,41)]
    scores_list = []

    # get the stopwords
    stop = set(stopwords.words('english'))

    for index, row in feature_file_df.iterrows():
        # get the json object of the news articles
        jsonObj_article_file = get_json_article_file(int(row['id']))

        # get the title of the news article
        jsonObj_article_content = get_jsonObj_article_content(row['id'])

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
        imgs_content = stem_imgs_content(imgs_content, stemer)
        imgs_content.append(query_string)

        # additional normalization: if keywords is a substring in image content, use the keyword as the cannonical form
        imgs_content, query_string = normalize_words(imgs_content, query_string)

        new_row_data = [row['id'], row['commons_url'], row['avg_rank'], score]
        output[int(row['id'])].append(new_row_data)

