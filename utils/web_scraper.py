import requests 
from bs4 import BeautifulSoup
import re
import string
import sys
import argparse

parser = argparse.ArgumentParser(description='Scrape an arabic poet\'s work !')
parser.add_argument('--poet', type=int,
                    default=0,
                    help='choosing the number of the poet')

args = parser.parse_args()

print(args.poet)


poet_number = args.poet

poets = ['mahmoud-darwish' ,'ahmed-shawqi', 'elia-abu-madi', 'Mutanabi', 'albohtry', 'abu-nawas']

print(poets[poet_number])
File = poets[poet_number]+".txt"


f = open(File, 'w+', encoding="utf-8")

def ScrapePoems (URL):
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="poem_content")
    #job_elements = results.find_all("div", class_="card-content")
    Poems = results.find_all("h3")
    for poem in Poems:
        f.write(poem.text.strip()+'\n')
    Poems = results.find_all("h4")
    for poem in Poems:
        f.write(poem.text.strip()+'\n')

BaseURL = "https://www.aldiwan.net/"
AuthURL = "https://www.aldiwan.net/cat-poet-"+poets[poet_number]
print(AuthURL)
page = requests.get(AuthURL)
soup = BeautifulSoup(page.content, "html.parser")
allTab = soup.find_all("div", class_="record col-12")
for tab in allTab:
    htmlUrl = tab.find("a", class_="float-right")['href']
    print(htmlUrl)
    ScrapePoems(BaseURL+htmlUrl)

f.close()


#Partie clean


punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation


arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

def preprocess(text):
    
    '''
    text is an arabic string input
    
    the preprocessed text is returned
    '''
    
    #remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)
    
    # remove Tashkeel
    text = re.sub(arabic_diacritics, '', text)
    
    #remove longation
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)


    return text

f = open(File, 'r', encoding="utf-8")
dataset = f.read()
cleaned = open(poets[poet_number]+'-clean.txt', 'w+',encoding="utf-8")
cleaned.write(preprocess(dataset))
f.close()
cleaned.close()