"""
This script is a collection of helper functions that will convert the bible search data into the correct shape 
for JSON objects to become D3.js, Plotly, or other open source visualization tools on the front end to
capture and output.
"""
from collections import Counter
import random

def htmlcolor(r, g, b):
    def _chkarg(a):
        if isinstance(a, int): # clamp to range 0--255
            if a < 0:
                a = 0
            elif a > 255:
                a = 255
        elif isinstance(a, float): # clamp to range 0.0--1.0 and convert to integer 0--255
            if a < 0.0:
                a = 0
            elif a > 1.0:
                a = 255
            else:
                a = int(round(a*255))
        else:
            raise ValueError('Arguments must be integers or floats.')
        return a
    r = _chkarg(r)
    g = _chkarg(g)
    b = _chkarg(b)
    return '#{:02x}{:02x}{:02x}'.format(r,g,b)

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
	colors=[]
	for _,_ in d3.items():
		r1=random.randint(0,255)
		r2=random.randint(0,255)
		r3=random.randint(0,255)
		color=htmlcolor(r1,r2,r3)
		colors.append(color)

	pie_output= {'type':'pie',
            	 'data':{'labels':list(d3.keys()),
                      	 'datasets':[{'data':list(d3.values()),
                      	 			  'backgroundColor':colors}],
                 'name': 'Books returned for: ' + searchText,
                 'hoverinfo':'label+percent+name'},
            	 'layout': {'title': "Book Analysis"}
            	 }
	print(pie_output)
	return pie_output
