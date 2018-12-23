"""
This script is a collection of helper functions that will convert the bible search data into the correct shape 
for JSON objects to become D3.js, Plotly, or other open source visualization tools on the front end to
capture and output.
"""
from collections import Counter
import random
from gensim.summarization import keywords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import math

colors=None
agg_labels=None
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

def bar_chart_reshape(searchText, data_dict):

	"""
	bar_chart_data= {type: 'bar',
					  data: {
					    labels: ["January", "February", "March", "April", "May", "June"],
					    datasets: [{
					      label: "Revenue",
					      backgroundColor: "rgba(2,117,216,1)",
					      borderColor: "rgba(2,117,216,1)",
					      data: [4215, 5312, 6251, 7841, 9821, 14984],
					    }],
					  },
					  options: {
					    scales: {
					      xAxes: [{
					        time: {
					          unit: 'month'
					        },
					        gridLines: {
					          display: false
					        },
					        ticks: {
					          maxTicksLimit: 6
					        }
					      }],
					      yAxes: [{
					        ticks: {
					          min: 0,
					          max: 15000,
					          maxTicksLimit: 5
					        },
					        gridLines: {
					          display: true
					        }
					      }],
					    },
					    legend: {
					      display: false
					    }
					  }
					});
	"""

	l=list(map(lambda x:" ".join(x.split(" ")[:-1]), list(data_dict.keys())))
	d2=dict(Counter(l))
	d3=dict(sorted(d2.items(), key = lambda x: x[1], reverse=True))

	bar_output= {'type':'horizontalBar',
	             'data':{'labels':agg_labels,
	                     'datasets':[{'label': "Occurances",
	                     			  'data':list(d3.values()),
	                                  'backgroundColor':colors}]},
	             'options': {
	                'scales': {
	                  'xAxes': [{
	                    'gridLines': {
	                      'display': False
	                    },
	                    'ticks': {
	                      'maxTicksLimit': len(list(d3.keys())),
	                    }
	                  }],
	                  'yAxes': [{
	                    'ticks': {
	                      'min': 0,
	                      'max': max(list(d3.values())),
	                      'maxTicksLimit': max(list(d3.values()))
	                    },
	                    'gridLines': {
	                      'display': True
	                    }
	                  }],
	                },
	                'legend': {
	                  'display': False
	                },
	                'responsive': True,
	   
		            'title': {
		                'display': True,
		                'text': searchText
		            },
		            'animation': {
		                'animateScale': True,
		                'animateRotate': True
		            }
	              }
	            }
	return bar_output

def keywords_reshape(searchText, data_dict):
	"""
	Return a list of overarching topics
	"""
	running_kwds=[]
	for k,v in data_dict.items():
	    running_kwds = running_kwds + keywords(v, split=True)

	return list(set(running_kwds))

def scatter_chart_reshape(searchText, data_dict):
	"""
		scatterData = {
	    datasets: [{
	      borderColor: 'rgba(99,0,125, .2)',
	      backgroundColor: 'rgba(99,0,125, .5)',
	      label: 'V(node2)',
	      data: [{
	        x: 25.1,
	        y: -5.429,
	      }, {
	        x: 31.6,
	        y: -6.944,
	      }]
	    }]
	  }

	  var config1 = new Chart.Scatter(ctxSc, {
	    data: scatterData,
	    options: {
	      title: {
	        display: true,
	        text: 'Scatter Chart - Logarithmic X-Axis'
	      },
	      scales: {
	        xAxes: [{
	          type: 'logarithmic',
	          position: 'bottom',
	          ticks: {
	            userCallback: function (tick) {
	              var remain = tick / (Math.pow(10, Math.floor(Chart.helpers.log10(tick))));
	              if (remain === 1 || remain === 2 || remain === 5) {
	                return tick.toString() + 'Hz';
	              }
	              return '';
	            },
	          },
	          scaleLabel: {
	            labelString: 'Frequency',
	            display: true,
	          }
	        }],
	        yAxes: [{
	          type: 'linear',
	          ticks: {
	            userCallback: function (tick) {
	              return tick.toString() + 'dB';
	            }
	          },
	          scaleLabel: {
	            labelString: 'Voltage',
	            display: true
	          }
	        }]
	      }
	    }
	  });

	  { type: 'scatter',
    data: {
        datasets: [{
            label: 'Scatter Dataset',
            data: [{
                x: -10,
                y: 0
            }, {
                x: 0,
                y: 10
            }, {
                x: 10,
                y: 5
            }]
        }]
    },
    options: {
        scales: {
            xAxes: [{
                type: 'linear',
                position: 'bottom'
            }]
        }
    }
});
	"""
	
	
	documents=list(map(lambda x:" ".join(x.split(" ")[:-1]), list(data_dict.keys())))
	text=list(data_dict.values())
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(text)
	pca = PCA(n_components=2)
	X_mapped = pca.fit_transform(X.todense())
	data=[]
	i=math.floor(len(X_mapped)/3)
	for point in X_mapped:
	    data.append({'x':point[0], 'y':point[1], 'r':i})
	    i-=1

	updated_colors=[]
	for d in documents:
		for i in range(len(agg_labels)):
			if d == agg_labels[i]:
				updated_colors.append(colors[i])

	ranks=[i for i in range(len(documents))].reverse()
	scatter_output= {	'type': 'scatter',
								#'bubble',
					    'data': {
					    		'label': "Proximity: ",
					    		'labels':list(data_dict.keys()),
						        'datasets': [{
						            'data': data,
						            'pointBackgroundColor':updated_colors
						        }],
						    },
					    'options': {
					    	'title':{'display':True,
			                         'text':searchText},
					        'scales': {
					            'xAxes': [{
					                'type': 'linear',
					                'position': 'bottom'
					            }]
					        },
					        'legend': {
					      				'display': False
					    			},
					    	'tooltips':{
					    		'callbacks':{
					    			'label':""
					    		}
					    	}
					    }
					};
	return scatter_output

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
	global colors
	global agg_labels

	colors=[]
	for _,_ in d3.items():
		r1=random.randint(0,255)
		r2=random.randint(0,255)
		r3=random.randint(0,255)
		color=htmlcolor(r1,r2,r3)
		colors.append(color)
	agg_labels=list(d3.keys())

	pie_output= {'type':'doughnut',
            	 'data':{#'label': "Percentage: ",
            	 		 'labels':agg_labels,
                      	 'datasets':[{'data':list(d3.values()),
                      	 			  'backgroundColor':colors}]},
            	 'responsive': True,
            'options': {
	            'responsive': True,
	   
	            'title': {
	                'display': True,
	                'text': searchText
	            },
	            'animation': {
	                'animateScale': True,
	                'animateRotate': True
	            },
	            'tooltips':{
					    		'callbacks':{
					    			'label':""
					    		}
					    	}

        }
            	 }
	return pie_output
