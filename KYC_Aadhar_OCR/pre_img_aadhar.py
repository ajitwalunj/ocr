import pytesseract
import urllib
import cv2
import numpy as np
from PIL import Image
import io
import re
import difflib
import csv
import dateutil.parser as dparser
from PIL import Image
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
import ftfy

import matplotlib.pyplot as plt
import os
import os.path
import json
import sys

class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

def process_image(url=None,path=None):
    
    if url != None:
    	image = url_to_image(url)
    elif path != None:
    	image = MyImage(path)
    else:
    	return "Wrong Wrong Wrong, What are you doing ??? "

    gray = cv2.cvtColor(image.img,cv2.COLOR_BGR2GRAY)
       #print ("Recognizing...")
    text=pytesseract.image_to_string(gray)
    name = None
    gender = None
    ayear = None
    uid = None
    yearline = []
    genline = []
    nameline = []
    text0 = []
    text1 = []
    text2 = []

    lines = text
    
    for wordlist in lines.split('\n'):
        xx = wordlist.split()
        if [w for w in xx if re.search('(Year|Birth|irth|YoB|YOB:|DOB:|DOB|DO8:|DO8|D08:|DOR:)$', w)]:
            yearline = wordlist
            break
        else:
            text1.append(wordlist)
    try:
        text2 = text.split(yearline, 1)[1]
    except Exception:
        pass


    try:
        yearline = re.split('Year|Birth|Birth |Birth :|Birth:|irth|YoB|DOB :|DOB:|DOB|DO8:|DO8 |D08:|DOR:', yearline)[1:]
        yearline = ''.join(str(e) for e in yearline)
        if(yearline):
            ayear = dparser.parse(yearline,fuzzy=True).year
    except:
        pass


    lineno = 0  

    for wordline in text1:
        xx = wordline.split('\n')
        if ([w for w in xx if re.search('(Government of India|vernment of India|overnment of India|ernment of India|India|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|GOVERNMENT OF INDIA|OVERNMENT OF INDIA|INDIA|NDIA)$', w)]):
            text1 = list(text1)
            lineno = text1.index(wordline)
            break

    text0 = text1[lineno+1:]
  

    try:
        for wordlist in lines.split('\n'):
            xx = wordlist.split( )
            if ([w for w in xx if re.search('(Female|Male|emale|male|ale|FEMALE|MALE|EMALE)$', w)]):
                genline = wordlist
                break
            
        if 'Male' in genline or 'MALE' in genline:
            gender = "Male"
    
        if 'Female' in genline or 'FEMALE' in genline:
            gender = "Female"
    

        text2 = text.split(genline,1)[1]

    except:
        pass

    text3= re.sub('\D', ' ', text2) 
    text3=text3.replace(" ","")
    text3=text3.replace("  ","")
    text3=text3.replace("   ","")
    text3=text3.replace("    ","")
    text3=text3.replace("     ","")
    text3[0:12]
    no=text3[0:12]

    #print(no)
    while("" in text0) : 
        text0.remove("") 
    while(" " in text0) : 
        text0.remove(" ") 
    while("  " in text0) : 
        text0.remove("  ") 
    while("   " in text0) : 
        text0.remove("   ") 
        

    name=text0[len(text0)-1]
    name = name.replace('|', "")
    name = name.replace('©)', "")
    name = name.replace('-',"")
    name = name.replace('!',"")
    
    DOB=yearline
    Uid=no
    Gender=gender
    Name=name
    
    data = {}
    data['Uid']=Uid
    data['Gender'] = gender
    data['Date of Birth'] = DOB
    data['Name'] = name

    data['Name'] = re.sub('[\W_]+', ' ', data['Name'])
    data['Gender'] = re.sub('[\W_]+', ' ', data['Gender'])
    data['Uid'] = re.sub('[\W_]+', ' ', data['Uid'])

    
    

    
    try:
            to_unicode = unicode
    except NameError:
            to_unicode = str


    with io.open(str(image) + '_data' +'.json', 'w', encoding='utf-8') as outfile:
            str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))



    print ("the result is {}".format(data))
    return data


	
