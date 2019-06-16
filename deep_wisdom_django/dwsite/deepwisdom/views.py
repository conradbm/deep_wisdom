from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from django.shortcuts import render_to_response
from django.core.mail import send_mail
from django.views.decorators.csrf import csrf_exempt

from DeepWisdom import DeepWisdom, get_db_connection
from core.plotting.plotting_transformers import pie_chart_reshape, bar_chart_reshape, scatter_chart_reshape, themes_reshape
from core.chat.ChatBot import chat_responses_reshape

import json
import sqlite3
import os
import datetime

DW=DeepWisdom()

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def get_query_results(search_query, ip, debug=True):
	if DW is None:
		print("Model not finished loading.")
		#text="<h1>Model not finished loading yet ...</h1></div><br><br>"
		#with open("templates/index.html", 'r') as handle:
		#	text+=handle.read().replace('\n','')
		#return text
		return {}
	else:
		print("Getting Search Results")
		searchText=search_query
		search_dict=DW.query(searchText, ip)
		#Consider json object being returned here.
		#results_string="<br>".join(["<strong>"+i[0]+"</strong>"+ " "+i[1] for i in result_tuples])
		
		if debug:
			print("Getting Pie results")
		pie_dict=pie_chart_reshape(searchText, search_dict)
		if debug:
			print("Getting Bar results")
		bar_dict=bar_chart_reshape(searchText, search_dict)
		if debug:
			print("Getting Scatter results")
		scatter_dict=scatter_chart_reshape(searchText, search_dict)
		if debug:
			print("Getting Themes results")
		themes_list=themes_reshape(searchText, search_dict)
		if debug:
			print("Getting ChatBot results")
		chat_dict_agg=chat_responses_reshape(searchText, search_dict)

		results_dict={'search_results':search_dict,
					  'pie_results':pie_dict,
					  'bar_results':bar_dict,
					  'scatter_results':scatter_dict,
					  'keyword_results':themes_list,
					  'chat_results_agg':chat_dict_agg}

		if debug:
			print("Search results")
			print(results_dict["search_results"])
			print("Pie results")
			print(results_dict["pie_results"])
			print("Bar results")
			print(results_dict["bar_results"])
			print("Scatter results")
			print(results_dict["scatter_results"])
			print("Keyword results")
			print(results_dict["keyword_results"])
			print("Chat results")
			print(results_dict["chat_results_agg"])
	return results_dict

def get_db_connection(root_dir="",loc=os.path.join("data","history.db")):
    """ DATABASE CONNECTION """
    # connect
    print("Connecting to database.")
    conn=sqlite3.connect(os.path.join(root_dir,loc) )
    if conn is not None:
        pass
    else:
        print("Error! cannot create the database connection.")
    return conn

def handle_login(requestPOST, ip, verbose=False):
	
	# Connect
	BASE_DIR=os.getcwd()
	conn=get_db_connection(root_dir=os.path.join( os.path.dirname( __file__ ), '..' ))

	# Sanitize
	email=requestPOST['email'].replace(";","_")
	password=requestPOST['password'].replace(";","_")
	time = str(datetime.datetime.now())
	
	results=conn.cursor().execute("SELECT * FROM T_User WHERE email=? AND password=?",(email, password)).fetchall()
	if len(results) == 0:
		print("User does not exist.")
		return {"success":False,"email":email,"reason":"This email password combination does not exist. {} {}".format(email,password)}

	# Debug
	if verbose:
		print(time)
		print(ip)
		print(requestPOST['email'])
		print(requestPOST['password'])

	# Insert
	conn.cursor().execute("""INSERT INTO T_Login (userid, time, ip) values (?,?,?) """,(results[0][0],time,ip))
	conn.commit()
	conn.close()

	# Example for indexing confusion
	# (1, '2019-06-03 15:48:46.566169', '127.0.0.1', 'blake', 'bmc.cs@outlook.com', 'pasSWoRd!!!'),
	# Send confirmation email
	send_mail(
	    'Hello, {}. Log in confirmation'.format(results[0][3]),
	    'Welcome back, {}. This is just a confirmation that you have logged back in at {} under {} email. Until next time,\n\n\n-Blake\nCEO DeepWisdom'.format(results[0][3], results[0][2], results[0][4]),
	    'deepwisdom.hq@gmail.com',
	    ['{}'.format(email)],
	    fail_silently=False,
	)
	return {"success":True, "id":results[0][0], "name": results[0][3]}


def reset():
	import sqlite3
	conn=sqlite3.connect('history.db')
	#drop_user_table="DROP TABLE T_User"
	#drop_login_table="DROP TABLE T_Login"
	create_user_table="CREATE TABLE IF NOT EXISTS T_User (userid integer PRIMARY KEY, time text, ip text, name text, email text, password text)"
	create_login_table="CREATE TABLE IF NOT EXISTS T_Login (loginid integer PRIMARY KEY, userid integer, time text, ip text)"
	#conn.cursor().execute(drop_user_table)
	conn.cursor().execute(create_user_table)
	#conn.cursor().execute(drop_login_table)
	conn.cursor().execute(create_login_table)
	r1=conn.cursor().execute("SELECT * FROM T_User").fetchall()
	r2=conn.cursor().execute("SELECT * FROM T_Login").fetchall()
	conn.commit()
	conn.close()
	return r1,r2 

def handle_register(requestPOST, ip, verbose=False):
	
	# Connect
	BASE_DIR=os.getcwd()
	conn=get_db_connection(root_dir=os.path.join( os.path.dirname( __file__ ), '..' ))

	# Sanitize
	name=requestPOST['name'].replace(";","_")
	email=requestPOST['email'].replace(";","_")
	password=requestPOST['password'].replace(";","_")
	time = str(datetime.datetime.now())
	
	# Verify this user doesn't already exist
	results=conn.cursor().execute("SELECT * FROM T_User WHERE email=?",(email,)).fetchall()
	print(results)
	if len(results) > 0:
		print("This email already exists in the database. Do not register them again.")
		return {'success':False,"name":name}

	# Debug
	if verbose:
		print(time)
		print(ip)
		print(requestPOST['name'])
		print(requestPOST['email'])
		print(requestPOST['password'])

	# Insert User
	conn.cursor().execute("""INSERT INTO T_User (time, ip, name, email, password) values (?,?,?,?,?)""",(time,ip,name,email,password))
	# Get UserId
	userid=conn.cursor().execute("""SELECT userid FROM T_User WHERE email=?""",(email,)).fetchall()[0][0]
	# Insert Login
	conn.cursor().execute("""INSERT INTO T_Login (userid, time, ip) values (?,?,?)""", (userid, time, ip))
	conn.commit()
	conn.close()

	# Send confirmation email
	send_mail(
	    'Hello, {}. Welcome to DeepWisdom!'.format(name),
	    'Thank you for joining us on your pursuit of wisdom, {}. This is confirmation of your new account with us with the following credentials: \nusername: {}\npassword: {}.\n\nWe are as excited about your pursuit of wisdom as you are, so we will keep you informed with all of the latest technology as it emerges. Until then,\n\n\n-Blake\nCEO DeepWisdom'.format(name, email, password),
	    'deepwisdom.hq@gmail.com',
	    ['{}'.format(email)],
	    fail_silently=False,
	)

	return {'success':True,"name":name}

@csrf_exempt
def index(request):

	ip = get_client_ip(request)
	# If registration request
	if request.method == 'POST':
		if 'name' in request.POST:
			
			resp=handle_register(request.POST, ip)
			return render_to_response('index.html')	

		# If sign in request
		elif 'email' in request.POST and 'password' in request.POST and not 'name' in request.POST:
			resp=handle_login(request.POST, ip)
			return render_to_response('index.html')	
	
	else:
		return render_to_response('index.html')	

@csrf_exempt
def charts(request):

	ip = get_client_ip(request)
	print(ip)
	context={}
	# If search request placed, return JSON
	if request.method == 'GET':
		# If search request
		if 'search' in request.GET:
			search_query = request.GET['search']
			results_dict=get_query_results(search_query, ip)
			json_data=json.dumps(results_dict)
			return HttpResponse(json_data, content_type="application/json")
		# Render normal
		else:
			return render_to_response('charts.html',context)
	# If registration request
	if request.method == 'POST':
		if 'name' in request.POST:
			resp=handle_register(request.POST, ip)

			return HttpResponse(json.dumps(resp), content_type="application/json")

		# If sign in request
		elif 'email' in request.POST and 'password' in request.POST and not 'name' in request.POST:
			resp=handle_login(request.POST, ip)

			return HttpResponse(json.dumps(resp), content_type="application/json")
	else:
		return render_to_response('charts.html',context)

		

