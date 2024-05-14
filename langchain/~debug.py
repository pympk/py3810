# Basics of recursion in Python
# https://stackoverflow.com/questions/30214531/basics-of-recursion-in-python
# https://stackoverflow.com/questions/39046311/python-base-case-of-a-recursive-function

# https://stackoverflow.com/questions/50270232/scrape-all-of-sublinks-of-a-website-recursively-in-python-using-beautiful-soup
# https://stackoverflow.com/questions/75539128/how-can-i-scrape-website-hyperlinks-in-a-recursive-way
# https://stackoverflow.com/questions/50270232/scrape-all-of-sublinks-of-a-website-recursively-in-python-using-beautiful-soup  

from bs4 import BeautifulSoup
import requests
from pyrsistent import v
from urllib.parse import urlparse


required_string = "https://www.lumenoptometric.com"
ignored_string = ".jpg"

# def write_to_file(links):
#   with open('data.txt', 'a') as f:
#     f.writelines(links)
def write_to_file(links):
  """
  Writes a list of links to a text file, adding a line return after each link.

  Args:
      links: A list of strings representing the links to write.
  """

  with open('data.txt', 'a') as f:
    for link in links:
      f.write(link)  # Add newline after each link
    f.write("\n")  # Add newline after each link  


unique_hrefs = set()

def get_website_links(url, visited=set(), iteration=0, verbose=False):
  """Fetches the HTML content of a website, extracts unique links, and recursively follows them."""

  # unique_hrefs = set()
  
  if iteration == 0:
    print(f'iter: {iteration}, url: {url}')
  else:
    visited.add(url)  # Mark current URL as visited
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all('a')
    # print(f'links: {links}')

    for link in links:
      print(f'link: {link}')
      href = link.get('href')
      if href and href not in visited:
        if required_string in href and ignored_string not in href:          
          unique_hrefs.add(href)
          write_to_file(href)
          visited.add(href)
          print(f'added href: {href}')
        # Recursive call for unvisited links
          # get_website_links(href, visited.copy(), iteration=iteration-1)  # Pass a copy of visited set
          get_website_links(href, visited.copy(), iteration=iteration-1)  # Pass a copy of visited set          

    if verbose:
      print(f'Links found on {url}:')
      for href in unique_hrefs:
        print(href)

  return unique_hrefs, visited

# Example usage
starting_url = "https://www.lumenoptometric.com"  # Replace with your target website
write_to_file([starting_url])
_uni_urls, _visited_urls = get_website_links(starting_url, iteration=4, verbose=False)
