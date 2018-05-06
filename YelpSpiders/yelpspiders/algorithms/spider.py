import urllib3
from bs4 import BeautifulSoup
import json
import re
import logging
import os
import time

from yelpspiders.variables.paths import Links, Path
from yelpspiders.utils.operations import get_number_of_pages


class Spider:

    def __init__(self, start_url, max_pages=-1, override=False):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.start_url = start_url
        self.http = urllib3.PoolManager()
        self.user_agent = {'user-agent': 'Mozilla/5.0'}
        self.max_pages = max_pages if max_pages > 0 else float('inf')
        self.override = override
        logging.info("Starting spiders...")

    def execute(self):
        response = self.http_request('GET', self.start_url)
        soup = BeautifulSoup(response.data, "lxml")
        pages = get_number_of_pages(soup)
        for p in range(1, min(pages + 1, self.max_pages + 1)):
            items = soup.find_all("li", {"class": "regular-search-result"})
            links = list(map(lambda x: x.find("a", {"class": "biz-name"}).get("href"), items))
            map(self.extract_data, links)
            response = self.http_request('GET', self.start_url + "&start=" + str(len(links) * p))
            soup = BeautifulSoup(response.data, "lxml")

    def extract_data(self, link):
        directory = "%s/%s" % (Path.DATA_FOLDER, link.replace("?frvs=True", "").replace("/biz/", ""))
        try:
            # JSON file
            json_file = dict()
            # JSON headers
            path = self.create_structure(directory)  # Raise Exception if the folder already exists
            json_file['path'] = path
            json_file['url'] = Links.BASE_LINK + link
            # HTTP to the restaurant url
            response = self.http_request('GET', Links.BASE_LINK + link)
            soup = BeautifulSoup(response.data, "lxml")
            # Location
            location = self.get_loc(soup)
            json_file['location'] = {'latitude': location[0], 'longitude': location[1]}
            # Menu
            menu = self.get_menu(soup)
            json_file['menus'] = menu
            # Show results
            with open('%s/info.json' % path, 'w') as fp:
                json.dump(json_file, fp)
            logging.info(json.dumps(json_file))
        except Exception as e:
            if "already exists" not in str(e):
                os.removedirs(directory)
            logging.error("%s at %s" % (e.message, str(link)))
        time.sleep(2)

    @staticmethod
    def create_structure(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            raise Exception("This folder already exists")
        return directory

    @staticmethod
    def get_loc(soup):
        latitude = re.findall("\"latitude\": [-]?\d+\.\d+", soup.get_text())[0].split()[1]
        longitude = re.findall("\"longitude\": [-]?\d+\.\d+", soup.get_text())[0].split()[1]
        return float(latitude), float(longitude)

    def get_menu(self, soup):
        menus_list = list()
        # URL of the menu
        menu_url = soup.find("a", {"class": "menu-explore"}).get("href")
        response = self.http_request('GET', Links.BASE_LINK + menu_url)
        soup = BeautifulSoup(response.data, "lxml")
        # Some restaurants could have some different menus
        menus = soup.find_all("li", {"class": "sub-menu"})
        other_menus = menus[1:] if len(menus) > 1 else list()
        try:
            # Main menu (current url)
            menus_list.append({'menu' if len(menus) < 2 else menus[0].getText().strip().lower(): self.get_menu_information(soup)})
        except Exception as e:
            logging.error("%s at %s" % (e.message, str(menus[0])))
        # Additional menus
        for m in other_menus:
            try:
                # New url
                menu_url = m.find('a').get("href")
                response = self.http_request('GET', Links.BASE_LINK + menu_url)
                soup = BeautifulSoup(response.data, "lxml")
                menus_list.append({m.getText().strip().lower(): self.get_menu_information(soup)})
            except Exception as e:
                logging.error("%s at %s" % (e.message, str(m)))
        return menus_list

    def get_menu_information(self, soup):
        menu_info = list()
        # Menu sections
        menu = soup.find("div", {"class": "menu-sections"})
        # Sections titles
        headers = menu.find_all("div", {"class": "section-header"})
        # Sections dishes
        info = menu.find_all("div", {"class": "u-space-b3"})
        info = [x for x in info if "menu-item" in str(x)]
        # Assign dishes to every section
        if len(headers) == len(info):
            for i in range(len(headers)):
                names_list, header = dict(), headers[i].find("h2").getText().strip().lower()
                for d in info[i].find_all("div", {"class": "menu-item-details"}):
                    try:
                        name, images = self.get_dish_images(d)
                        names_list[name] = images
                    except Exception as e:
                        logging.error("%s at %s" % (e.message, str(d)))
                menu_info.append({header: names_list})
        else:
            raise Exception("Headers and/or Items are not formatted as expected")
        return menu_info

    def get_dish_images(self, soup):
        # Name
        name = soup.find("h4").getText().strip().lower()
        # Get images URL
        d = soup.find("div", {"class": "menu-item-details-stats"})\
            .find("a", class_=lambda x: x != 'num-reviews')
        href = d.get("href")
        # Go to photos
        response = self.http_request('GET', Links.BASE_LINK + href)
        soup = BeautifulSoup(response.data, "lxml")
        photos_url = soup.find("a", {"class": "more-photos-link"}).get("href")
        response = self.http_request('GET', Links.BASE_LINK + photos_url)
        soup = BeautifulSoup(response.data, "lxml")
        # URL for images
        urls_images = self.get_images_list(Links.BASE_LINK + photos_url, soup)
        return name, urls_images

    def get_images_list(self, photos_url, soup):
        pages = get_number_of_pages(soup)
        # URLS
        urls = list()
        for p in range(1, pages + 1):
            items = soup.find_all("div", {"class": "photo-box"})
            # Remove the first link because it is the root
            links = list(map(lambda x: x.find("img").get("src"), items))[1:]
            for l in links:
                try:
                    urls.append(l.replace(l[l.rfind("/"): len(l)], "/o.jpg"))
                except Exception as e:
                    logging.error("%s at %s" % (e.message, str(l)))
            response = self.http_request('GET', photos_url + "&start=" + str(len(links) * p))
            soup = BeautifulSoup(response.data, "lxml")
        return urls

    def http_request(self, param, start_url):
        return self.http.request(param, start_url, headers=self.user_agent)


if __name__ == '__main__':
    s = Spider('https://www.yelp.com/search?find_loc=LasVegas,+CA&cflt=restaurants', max_pages=100)
    s.execute()
