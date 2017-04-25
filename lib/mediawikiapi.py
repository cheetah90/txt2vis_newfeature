import requests
from bs4 import BeautifulSoup

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