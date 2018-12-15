from flask import request
from app import app
import os
from . import DeepWisdom
from threading import Thread
from flask import render_template, json

# Helper for killing python processes
# pkill -f *.py

"""
Initialization of model data
"""
initialized=False

DW=None
def create_class(args):
	# Once the site is
	global DW
	DW=DeepWisdom.DeepWisdom()

def async_create_class(args):
    thr = Thread(target=create_class, args=[args])
    thr.start()
    return thr

"""
Static HTML pages the user will get from navigating the site
"""
@app.route('/')
@app.route('/about.html')
def index():

	global initialized
	if not initialized:
		# Load the data in another thread
		async_create_class([])
		initialized=True

    # Serve up home page
	with open("app/templates/about.html", 'r') as handle:
		text=handle.read().replace('\n','')
	return text

@app.route('/contact')
@app.route('/contact.html')
def contact():
	with open("app/templates/contact.html", 'r') as handle:
		text=handle.read().replace('\n','')
	return text

@app.route('/explore')
@app.route('/explore.html')
def explore():
	with open("app/templates/explore.html", 'r') as handle:
		text=handle.read().replace('\n','')
	return text


"""
Functions from forms and other widgets
"""


##### LEFT OFF HERE #####

@app.route('/submit', methods=['POST'])
def submit():
	if DW is None:
		print("Model not finished loading.")
		with open("app/templates/explore.html", 'r') as handle:
			text=handle.read().replace('\n','')
		return "<div class='jumobtron'><h1>Model not finished loading yet ...</h1></div><br><br>"+text 
	else:
		print("Searching")
		conn=DeepWisdom.get_db_connection()
		searchText=request.form['search']
		results=DW.query(conn, searchText)
		results_string="<br>".join(["<strong>"+i[0]+"</strong>"+ " "+i[1] for i in results])
		return results_string
		
		#return json.dumps(results_string)
