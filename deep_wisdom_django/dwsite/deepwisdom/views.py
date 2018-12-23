from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from django.shortcuts import render_to_response
import json
from DeepWisdom import DeepWisdom, get_db_connection
from core.plotting.plotting_transformers import pie_chart_reshape, bar_chart_reshape, scatter_chart_reshape, keywords_reshape

DW=DeepWisdom()

def get_query_results(search_query):
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
		search_dict=DW.query(searchText)
		#Consider json object being returned here.
		#results_string="<br>".join(["<strong>"+i[0]+"</strong>"+ " "+i[1] for i in result_tuples])
		print("Getting Pie Results")
		pie_dict=pie_chart_reshape(searchText, search_dict)
		print("Getting Bar Results")
		bar_dict=bar_chart_reshape(searchText, search_dict)
		print("Getting Scatter results")
		scatter_dict=scatter_chart_reshape(searchText, search_dict)
		print("Getting Keyword results")
		keywords_list=keywords_reshape(searchText, search_dict)

		results_dict={'search_results':search_dict,
					  'pie_results':pie_dict,
					  'bar_results':bar_dict,
					  'scatter_results':scatter_dict,
					  'keyword_results':keywords_list}
		print("Search results")
		print(results_dict["search_results"])
		print("Pie Results")
		print(results_dict["pie_results"])
		print("Bar Results")
		print(results_dict["bar_results"])
		print("Scatter results")
		print(results_dict["scatter_results"])
		print("Keyword results")
		print(results_dict["keyword_results"])
		return results_dict

def index(request):
    return render_to_response('index.html')	

def charts(request):

	context={}
	# If search request placed, return JSON
	if request.method == 'GET':
		if 'search' in request.GET:
			search_query = request.GET['search']
			print(search_query, "Working!")
			results_dict=get_query_results(search_query)
			json_data=json.dumps(results_dict)
			return HttpResponse(json_data, content_type="application/json")

		# Render normal
		else:
			return render_to_response('charts.html',context)

	