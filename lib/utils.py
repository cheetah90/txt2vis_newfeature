import nltk
from collections import Counter


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text, stemmer):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def stem_imgs_content(imgs_content, stemmer):
    stemmed_imgs_content = []

    for img_content_item in imgs_content:
        stemmed_tokens = tokenize(img_content_item, stemmer)
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


