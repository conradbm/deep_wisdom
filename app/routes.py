from flask import Flask
from flask import request
import os
from DeepWisdom import DeepWisdom, get_db_connection
from threading import Thread
from flask import render_template, json

# Helper for killing python processes
# pkill -f *.py

app = Flask(__name__)

"""
Initialization of model data
"""
initialized=False

DW=None
def create_class(args):
	# Once the site is
	global DW
	DW=DeepWisdom()

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

	return render_template("about.html")

@app.route('/contact')
@app.route('/contact.html')
def contact():
	return render_template("contact.html")

@app.route('/explore')
@app.route('/explore.html')
def explore():
	return render_template("explore.html")


"""
Functions from forms and other widgets
"""


##### LEFT OFF HERE #####

@app.route('/submit', methods=['POST'])
def submit():
	if DW is None:
		print("Model not finished loading.")
		with open("templates/explore.html", 'r') as handle:
			text=handle.read().replace('\n','')
		return "<div class='jumobtron'><h1>Model not finished loading yet ...</h1></div><br><br>"
	else:
		print("Searching")
		conn=get_db_connection()
		searchText=request.form['search']
		results=DW.query(conn, searchText)
		results_string="<br>".join(["<strong>"+i[0]+"</strong>"+ " "+i[1] for i in results])
		return results_string
		
		#return json.dumps(results_string)

if __name__ == '__main__':
	app.run()