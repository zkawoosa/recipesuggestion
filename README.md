# RecipEASY's Recipe Finder

A recommendation search engine that will allow you to input a list of ingredients you have in your fridges and cupboards and will provide you with a ranked list of the Top 10 most relevant recipes you can make with your ingredients. A custom ranking algorithm is utilized to order the results.

## Getting Started
Required external libraries are: flask, match, BeautifulSoup4, numpy, numpy.linalg, fuzzywuzzy, sklearn, umap.

## Running Code

The server can be run with the command "python3 app.py", and the UI can be found at "http://localhost:5000/"

## Table of Contents 

templates/index.html:
    Jinja template which takes in retrieved recipes and associated information 
    and formats the output for the UI.

app.py:
    This file contains all of the flask routes implemented for the UI, 
    as well as functions to determine similar ingredients to the user's input 
    and calculate macro averaged precision and recall metrics. This file imports
    functions from match.py in order to complete these tasks. 

crawler.output:
    This file contains 10,000 URLs retrieved from the crawler, 
    where each URL represents one recipe. These URLs were then 
    inputted into the extractor. 

crawler.py:
    This file contains functions used to check URL domains, 
    check if the URL has been visited, and determine if the URL is a redirect. 
    The crawler uses all of these functions to check a valid, unvisited URL,
    collect all associated URLs, and then recursively check those URLs until 
    enough URLs were collected. Finally, these URLs are outputted to crawler.output.

extractor.py:
    This code reads in URLs from crawler.output, parses them using BeautifulSoup 
    and outputs recipe names mapped to ingredient lists and 
    ingredient lists mapped to recipe names to dishTOingredients.output and 
    ingredientTOdishes.output respectively. For efficiency purposes, 
    multithreading is used to parse 10 pages concurrently, 
    and the same session is used for all requests. 


dishTOingredients.output:
    This file contains a mapping of recipes names to their associated ingredients. 

ingredientTOdishes.output:
    This file contains a mapping of ingredient lists to their recipe names. 

match.py:
    This file contains functions used to retrieve a sorted list of recipes based 
    on a user's inputted ingredients. These functions perform many calculations,
    including levenshtein distance between strings, levenshtein distance 
    between lists of strings, fuzzy searching between stored ingredient lists 
    and user inputted ingredients in order to determine the set of relevant recipes, 
    and three search functions which calculate similarity scores based on levenshtein 
    distance, cosine similarity with dimensional reduction, cosine similarity, and a
    custom similarity score in which Jaccard similarity is calculated and normalized 
    by the difference in the number of ingredients found in a recipe vs the number of 
    ingredients inputted by the user. 
    Finally, a flattening function is used to produce a flattened array of ingredients. 

starturls.txt:
    This file contains the base url at which the crawler began.




