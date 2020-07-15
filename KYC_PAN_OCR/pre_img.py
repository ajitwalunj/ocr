
#coding:utf-8
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
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
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


#	image = _get_image(url)
	if url != None:
		image = url_to_image(url)
	elif path != None:
		image = MyImage(path)
	else:
		return "Wrong Wrong Wrong, What are you doing ??? "

	gray = cv2.cvtColor(image.img,cv2.COLOR_BGR2GRAY)

	print ("Recognizing...")

	text =  pytesseract.image_to_string(gray,lang = 'eng')
	name = None
	fname = None
	dob = None
	pan = None
	nameline = []
	dobline = []
	panline = []
	text0 = []
	text1 = []
	text2 = []
	govRE_str = '(GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$'
	numRE_str = '(Number|umber|Account|ccount|count|Permanent|renmanent|manent|Per#ianent Account Number|Perdfighent Account Number|Penfianent Account Number)$'
	incOM_str = '(INCOMETAXDEPARWENT|INCOME|TAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA| INCOME TAX DEPARTMENT|TAX DEPARTMENT|INCOME TAX DEPARTMENT|INCOMETAK DEPARTMENT >  GOVT.OF INDIA -)$'
	text = ftfy.fix_text(text)
	text = ftfy.fix_encoding(text)


	lines = text.split('\n')
	for lin in lines:
		s = lin.strip()
		s = lin.replace('\n','')
		s = s.rstrip()
		s = s.lstrip()
		text1.append(s)

	text1 = list(filter(None, text1))

	lineno = -1  

	for wordline in text1:
		xx = wordline.split('\n')
		if ([w for w in xx if re.search(incOM_str, w)]):
			text1 = list(text1)
			lineno = text1.index(wordline)
			break

	text0 = text1[lineno+1:]

	def clean(textlist):
		clean_text = []
		for wordline in textlist:
			xx = wordline.split()
			if(any(map(str.isdigit, wordline)) or [w for w in xx if re.match('^(?=.*[a-zA-Z])', w)]):
				clean_text.append(wordline)
		return clean_text

	text2  = clean(text0)

	def get_DOB(textlist):
		for wordline in textlist:
			if(any(map(str.isdigit, wordline))):
				date = wordline
				break
		return date

	if(len(text2)<1):
		text2.append('Name Not Found')
		text2.append('Father Name Not Found')
		text2.append('00/00/00')
		text2.append('Permanent Account Number')
		text2.append('PAN Not Found')

	elif (len(text2)<2):
		text2.append(text2[0])
		text2.append(text2[0])
	elif (len(text2)<3):
		text2.append(text2[0])
		text2.append(text2[1])


	data = {}
	data['Name'] = text2[0]
	data['Father Name'] = text2[1]
	if(get_DOB(text2)):
		data['Date of Birth'] = get_DOB(text2)
	else:
		data['Date of Birth'] = text2[2]




	def findword(textlist, wordstring):
		lineno = -1
		for wordline in textlist:
			xx = wordline.split( )
			if ([w for w in xx if re.search(wordstring, w)]):
				lineno = textlist.index(wordline)
				textlist = textlist[lineno+1:]
				return textlist
		return textlist


	text3 = findword(text2, '(Pormanam|Number|umber|Account|ccount|count|Permanent|ermanent|manent|wumm|Per#ianent Account Number|Perdfighent Account Number|Penfianent Account Number)$')
	data['PAN'] = text3[0]


	data['Name'] = re.sub('[\W_]+', ' ', data['Name'])
	data['Father Name'] = re.sub('[\W_]+', ' ', data['Father Name'])
	data['PAN'] = re.sub('[\W_]+', ' ', data['PAN'])

	try:
		to_unicode = unicode
	except NameError:
		to_unicode = str


	with io.open(str(image) + '_data' +'.json', 'w', encoding='utf-8') as outfile:
		str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
		outfile.write(to_unicode(str_))


	print ("the result is {}".format(data))
	return data
