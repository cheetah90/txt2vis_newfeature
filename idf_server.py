import requests

class IDFserverAPI:
    def __init__(self):
        self.baseUrl = 'http://localhost:9200/edn_idf/'

    def getMinIDF(self, word):
        url = self.baseUrl + "_search?q=word:\"{}\"".format(word)
        response = requests.get(url)
        min_idf = 100
        matched_flag = False

        if (response.status_code == 200):
            response = response.json()
            if not response['timed_out']:
                for item in response['hits']['hits']:
                    if item['_source']['word'].lower() == word.lower() and item['_source']['idf'] < min_idf:
                        matched_flag = True
                        min_idf = item['_source']['idf']

        return min_idf if matched_flag else 0