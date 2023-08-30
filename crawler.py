import requests
from bs4 import BeautifulSoup
import sys
import re
import datetime

# crawler.py
# PURPOSE: Collects a variable amount of recipe URLs from allrecipes.com, prints each URL to crawler.output

# url_strip: removes http://, https://, and any trailing slashes.
# NOTE: if the url starts with only leading slashes, such as '//eecs.engin.umich.edu', these are
#       NOT stripped away. Since I do not consider this type of link to be valid as a design choice,
#       not stripping them away at this step in my program allows the url_validate() to flag this incorrectly
#       formatted URL at that stage.
def url_strip(url):
  # Removing any leading http:// or https://
  stripped_url = re.compile(r"https?://(www\.)?").sub("", url)
  # Removing trailing backslashes
  stripped_url = re.compile("\/$").sub("", stripped_url)
  return stripped_url

# url_visited: if the URL is in links_visited, return True. Otherwise, return False.
def url_visited(url, links_visited):
    if url_strip(url) in links_visited:
        return True
    else:
        return False

# url_validate: URL is stripped then checked to make sure the stripped URL begins with an acceptable domain.
def url_validate(url, accepted_domains):
  url = url_strip(url)
  for domain in accepted_domains:
    if url.startswith(domain):
      return True
  return False

# url_redirect_check: Sees if a URL redirects to another URL. If this is a case, we return (True, <final URL>), otherwise return (False, None).
def url_redirect_check(url):
    response = requests.head(url, allow_redirects=True)
    if response.history:
        return True, response.url
    else:
        return False, None


def main():
    initial_time = datetime.datetime.now()
    seed_url_file = sys.argv[1]
    max_pages = int(sys.argv[2])

    with open(seed_url_file, 'r') as f:
        start_urls = f.read().splitlines()

    ACCEPTED_DOMAINS = ['allrecipes.com/recipe/']

    links_queue = start_urls[:]

    # This list contains a list of all links that have recipes on them from the specified start domain
    links_identified = start_urls[:]

    # This list contains stripped versions of each link that has been identified, for the purposes of checking whether the link has been visited already or not.
    links_visited = []
    for url in start_urls:
        links_visited.append(url_strip(url))


    get_session = requests.Session()

    while len(links_identified) < max_pages:
        # Removing link from queue and adding it to links_identified
        parent_link = links_queue.pop(0)

        try:
            # Fetching HTML from webpage at parent_link
            response = get_session.get(parent_link, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
            })

            # Checking if page is HTML and the URL has an accepted domain
            parser = BeautifulSoup(response.content, "html.parser")
            links = parser.find_all('a')

            for link in links:
                    child_link = link.get('href')

                    # Making sure the child_link exists and isn't NoneType
                    if child_link:
                        # Checking if link is part of valid domain, hasn't been visited, and leads to a valid HTML page
                        if url_validate(child_link, ACCEPTED_DOMAINS):

                            if not url_visited(child_link, links_visited):
                                # Adding stripped version of link to a visited list
                                links_visited.append(url_strip(child_link))

                                # Adding identified link
                                links_identified.append(child_link)
                                print(len(links_identified))
                                print(child_link)

                                # Adding link to be downloaded
                                links_queue.append(child_link)

                            # Breaking out of loop once we've identified a variable amount of links
                            if len(links_identified) == max_pages:
                                break

        except requests.exceptions.RequestException as e:
            print("Error for fetching following link: ", parent_link)
            print("Error Message:")
            print(e)

    # Writing to output files
    with open('crawler.output', 'w') as f:
        for link in links_identified:
            if link != 'https://www.allrecipes.com':
                f.write(link + '\n')

    final_time = datetime.datetime.now()

    print("START TIME: ", initial_time)
    print("END TIME:   ", final_time)


if __name__ == '__main__':
    main()
