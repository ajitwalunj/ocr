import re

pattern = re.compile("^[a-zA-Z]+$")
s = 'Vaibhav Dubey'
if pattern.match(s):
	print('yes')