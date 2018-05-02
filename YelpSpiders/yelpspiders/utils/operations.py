from bs4 import BeautifulSoup


def get_number_of_pages(soup):
    pages = 0
    for l in soup.find_all("div", {"class": "page-of-pages"}):
        split = l.get_text().split()
        try:
            pages += int(split[len(split)-1])
        except:
            pass
    return pages
