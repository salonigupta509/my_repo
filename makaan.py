# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:47:40 2018

@author: user pc
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:47:40 2018

@author: user pc
"""


from  selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup as BS

url = "https://www.makaan.com/jaipur-residential-property/rent-property-in-jaipur-city"
browser = webdriver.Chrome("C:\Users\user pc\Downloads\chromedriver_win32\chromedriver.exe")
browser.get(url)

def btn_click(browser_con,num):
    if num>3:
        result = browser_con.find_element_by_xpath('//*[@id="mod-listingsWrapper-{0}"]/div/div[3]/div[1]/div/ul/li[{1}]/a'.format(num,9))
        result.click()
    else:
        result = browser_con.find_element_by_xpath('//*[@id="mod-listingsWrapper-{0}"]/div/div[3]/div[1]/div/ul/li[{1}]/a'.format(num,8))
        result.click()
    #sleep(10)
    
    try:
      html_page = browser_con.page_source
    
    except Exception: 
    #  pass 
        html_page = browser_con.page_source
    return html_page

sleep(5)
script = browser.find_element_by_class_name('overlay')
script.click()

for i in range(0,5):
    page = btn_click(browser,i+1)
    soup = BS(page,"lxml")



"""import urllib2
import numpy as np
from bs4 import BeautifulSoup as bs
url = "https://www.makaan.com/jaipur-residential-property/rent-property-in-jaipur-city"
page=urllib2.urlopen(url)
soup=bs(page,"lxml")"""
all_div = soup.find_all('div',class_ = "infoWrap")

location=[]
area=[]
BHK=[]
price = []

#BHK
for section in all_div:
    info = section.find_all('div',class_="title-line")
    n_raw = info[0].text.strip()
    n_raw=int(n_raw[0])
    BHK.append(n_raw) 
    
#location
for section in all_div:
    info = section.find_all('div',class_="locWrap")
    n_raw = info[0].text.strip()
    n_raw=n_raw.split(',')
    n_raw=n_raw[0]
    location.append(n_raw)

#price
for section in all_div:
    info = section.find_all('div',class_="price")
    n_raw=info[0].text.strip().replace(',','')
    n_raw=int(n_raw)
    price.append(n_raw)
    
#area
for section in all_div:
    info = section.find_all('div',class_="size")
    n_raw=info[0].text.strip()
    area.append(n_raw)
    
#Making the dataframe
import pandas as pd

df1 = pd.DataFrame(location,columns = ['Location'])
df1['Area'] = area
df1['BHK'] = BHK
df1['price'] = price
df1.to_csv("makan1.csv",index=False)
    
    

