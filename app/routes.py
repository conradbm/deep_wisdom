import os
from flask import Flask
from flask import request
from DeepWisdom import DeepWisdom, get_db_connection
from threading import Thread
from flask import render_template, jsonify
from core.plotting.plotting_transformers import pie_chart_reshape

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
@app.route('/index.html')
def index():

	global initialized
	if not initialized:
		# Load the data in another thread
		async_create_class([])
		initialized=True

	return render_template("index.html")

@app.route('/charts')
@app.route('/charts.html')
def charts():
	return render_template("charts.html")

@app.route('/tables')
@app.route('/tables.html')
def tables():
	return render_template("tables.html")


"""
Functions from forms and other widgets
"""
@app.route('/submit', methods=['POST'])
def submit():
	if DW is None:
		print("Model not finished loading.")
		#text="<h1>Model not finished loading yet ...</h1></div><br><br>"
		#with open("templates/index.html", 'r') as handle:
		#	text+=handle.read().replace('\n','')
		#return text
		return render_template("index.html")
	else:
		print("Searching")
		conn=get_db_connection()
		searchText=request.form['search']
		search_dict=DW.query(conn, searchText)
		#Consider json object being returned here.
		#results_string="<br>".join(["<strong>"+i[0]+"</strong>"+ " "+i[1] for i in result_tuples])
		pie_dict=pie_chart_reshape(searchText, search_dict)

		results_dict={'search_results':search_dict,
					  'pie_results':pie_dict}
		return jsonify(results_dict)
		
		#return json.dumps(results_string)

if __name__ == '__main__':
	app.run()