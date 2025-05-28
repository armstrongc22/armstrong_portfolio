import requests, pandas as pd
from bs4 import BeautifulSoup

def fetch_wikipedia_reactors():
    url = "https://en.wikipedia.org/wiki/List_of_nuclear_reactors"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # find the right table, parse into df...
    tables = pd.read_html(r.text)
    reactors = tables[0]  # adjust index as needed
    reactors.to_csv("data/reactors.csv", index=False)
    return reactors
