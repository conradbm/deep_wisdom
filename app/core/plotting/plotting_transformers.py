"""
This script is a collection of helper functions that will convert the bible search data into the correct shape 
for JSON objects to become D3.js, Plotly, or other open source visualization tools on the front end to
capture and output.
"""
from collections import Counter
def pie_chart_reshape(searchText, data_dict):

	"""
	Objective of this function is to get the counts of each book that was returned from our query into a clean
	pie chart. This will be done by parsing our keys returned, splitting on space, and taking the first indice.

	INPUT EXAMPLE:
	data = {'Genesis 1:1': ['in the beginning god created the heavens and the earth', '0.18'],
			'John 1:1': ['in the beginning was the word and the word was with god and the word was god', '0.16'],
			...}

	OUTPUT EXAMPLE:
	pie_chart = {'data': [{
				    'labels': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9'],
				    'values': [55, 22, 31, 32, 33, 45, 44, 42, 12, 67],
				    'marker': {
				      'colors': [
				        'rgb(0, 204, 0)',
				        'rgb(255, 255, 0)',
				        'rgb(118, 17, 195)',
				        'rgb(0, 48, 240)',
				        'rgb(240, 88, 0)',
				        'rgb(215, 11, 11)',
				        'rgb(11, 133, 215)',
				        'rgb(0, 0, 0)',
				        'rgb(0, 0, 0)',
				        'rgb(0, 0, 0)'
				      ]
				    },
				    'type': 'pie',
				    'name': "Books Returned",
				    'hoverinfo': 'label+percent+name',
				    'sort': false,
				  }],

				  'layout': {
				    'title': 'Books Results Analysis'
				  }
				}

	
	Run the below in javascript to get the pie chart.
	Plotly.newPlot('plot', pie_chart.data, pie_chart.layout);
	"""

	l=list(map(lambda x:" ".join(x.split(" ")[:-1]), list(data_dict.keys())))
	d2=dict(Counter(l))
	d3=dict(sorted(d2.items(), key = lambda x: x[1], reverse=True))

	pie_output= {'data':[{'labels':list(d3.keys()),
	                      'values':list(d3.values()),
	                      'type':'pie',
	                      'hoverinfo':'label+percent',
	                      'sort':False}],
	            'layout': {'title': "Book Analysis: " + searchText}}
	return pie_output
