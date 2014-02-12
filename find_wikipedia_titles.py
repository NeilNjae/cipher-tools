import urllib.request
import urllib.parse
import json
import time

initial_request_url = "http://en.wikipedia.org/w/api.php?action=query&list=allpages&format=json&aplimit=10&apminsize=5000"
request_url = "http://en.wikipedia.org/w/api.php?action=query&list=allpages&format=json&aplimit=10&apminsize=5000&apcontinue={}"
titles_file = '/opt/sources/wp-titles.txt'

def titles_of(result):
    return [p['title'] for p in result['query']['allpages'] ]

def next_title(result):
    return result['query-continue']['allpages']['apcontinue']

def write_titles(titles):
    with open(titles_file, 'a') as f:
        print('\n'.join(titles), file=f)

def request_again(start_title):
    request = urllib.request.Request(request_url.format(urllib.parse.quote(start_title)))
    request.add_header('User-Agent','neil.wpspider@njae.me.uk')
    result = json.loads(urllib.request.urlopen(request).read().decode())
    return titles_of(result), next_title(result)

f = open(titles_file, 'w')
f.close()

result = json.loads(urllib.request.urlopen(initial_request_url).read().decode())
n_title = next_title(result)
titles = titles_of(result)
while titles != []:
    write_titles(titles)
    time.sleep(0.5)
    titles, n_title = request_again(n_title)
