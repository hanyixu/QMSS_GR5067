# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:05:36 2019

@author: pathouli
"""

def init():
    #https://pypi.org/project/torpy/
    #pip install torpy
    from torpy import TorClient
    hostname = 'ifconfig.me'  # It's possible use onion hostname here as well
    with TorClient() as tor:
        # Choose random guard node and create 3-hops circuit
        with tor.create_circuit(3) as circuit:
            # Create tor stream to host
            with circuit.create_stream((hostname, 80)) as stream:
                # Now we can communicate with host
                stream.send(b'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % hostname.encode())
                recv = stream.recv(1024)
    return 0

def my_scraper(tmp_url_in):
    #https://pypi.org/project/beautifulsoup4/
    #pip install beautifulsoup4
    from bs4 import BeautifulSoup
    import requests
    import re
    import time
    tmp_text = ''
    try:
        content = requests.get(tmp_url_in, timeout=10)
        soup = BeautifulSoup(content.text, 'html.parser')

        tmp_text = soup.findAll('p') 

        tmp_text = [word.text for word in tmp_text]
        tmp_text = ' '.join(tmp_text)
        tmp_text = re.sub('\W+', ' ', re.sub('xa0', ' ', tmp_text))
    except:
        print("Connection refused by the server..")
        print("Let me sleep for 5 seconds")
        print("ZZzzzz...")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")
        pass
    return tmp_text

def fetch_urls(query_tmp, cnt):
    #now lets use the following function that returns
    #URLs from an arbitrary regex crawl form google
    #pip install pyyaml ua-parser user-agents fake-useragent
    import requests
    from bs4 import BeautifulSoup
    import re 
    headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
    query = '+'.join(query_tmp.split())
    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(cnt)
    print (google_url)
    response = requests.get(google_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    result_div = soup.find_all('div', attrs = {'class': 'egMi0 kCrYT'})

    links = []
    links_t = []
    for r in result_div:
        # Checks if each element is present, else, raise exception
        try:
            links_t.append(r.a.get('href'))
            for link in links_t:
                x = re.split("&ved", (re.split("url=", link)[1]))
                if x[0] not in links:
                    links.append(x[0])
            if link != '':# and title != '' and description != '': 
                links.append(link['href'])
        # Next loop if one element is not present
        except:
            continue  
    return links
 
def write_crawl_results(my_query, the_cnt_in):
    #let use fetch_urls to get URLs then pass to the my_scraper function 
    import re
    import pandas as pd 

    tmp_pd = pd.DataFrame()       
    for q_blah in my_query:
        #init()
        the_urls_list = fetch_urls(q_blah, the_cnt_in)

        for word in the_urls_list:
            tmp_txt = my_scraper(word)
            if len(tmp_txt) != 0:
                try:
                    t = pd.DataFrame(
                        {'body': tmp_txt, 'label': re.sub(
                            ' ', '_', q_blah)}, index=[0])
                    tmp_pd = pd.concat([tmp_pd, t], ignore_index=True)
                    print (word)
                except:
                    pass
    return tmp_pd