# Vector Space Model
from random import random
import itertools
import numpy as np
from numpy.linalg import norm
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


# Return the levenshtein distance between strings a and b
def string_ldistance(a, b):
    # Create len(a) + 1 by len(b) + 1 matrix of zeros
    matrix = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]

    # Preset values of matrix before dp
    for i in range(len(a)+1):
        matrix[i][0] = i
    for j in range(len(b)+1):
        matrix[0][j] = j

    # Dynamically compute matrix values, return value at len(a), len(b)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = 1 + \
                    min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1])

    return matrix[len(a)][len(b)]


# Checks if all of the words that the user inputted are found in an ingredient in the InvertedIndex
def checkMatchingWords(a, b):
    aSet = set()
    bSet = set()
    aList = a.split()
    bList = b.split()
    for word in aList:
        if "," in word:
            wordList = word.split(",")
            for w in wordList:
                aSet.add(w)
        else:
            aSet.add(word)
    for word in bList:
        if "," in word:
            wordList = word.split(",")
            for w in wordList:
                bSet.add(w)
        else:
            bSet.add(word)

    # check for ingredients with "fuzzy matching"
    # if levenstein difference b/w ingredients is less than 3, then they are considered a match (allows for typos)
    for word in aSet:
        matched = False
        for ingredient in bSet:
            if string_ldistance(word, ingredient) < 3:
                matched = True
                break
        if not matched:
            return False
    return True


def searchRecipes(user_ingredients):
    # map of ingredient to list of recipes that need that ingredient
    invertedIndex = {}
    # map of recipes to their list of ingredients
    recipeToIngredients = {}
    with open('dishTOingredients.output', "r") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if len(line):
                # build inverted index
                recipe = line.split(":")[0].strip()
                allIngredients = line.split(":")[1].strip()
                ingredientsList = allIngredients.split("|")
                for ingredient in ingredientsList:
                    if ingredient in invertedIndex:
                        invertedIndex[ingredient].append(recipe)
                    else:
                        invertedIndex[ingredient] = [recipe]
                    if recipe in recipeToIngredients:
                        recipeToIngredients[recipe].add(ingredient)
                    else:
                        recipeToIngredients[recipe] = set([ingredient])

            line = fp.readline()

    # set of all recipes that have at least one ingredient from the user inputted list of ingredients
    recipeSet = set()
    for ingredient in user_ingredients:
        for key in invertedIndex:
            if checkMatchingWords(ingredient, key):
                recipeSet = recipeSet.union(set(invertedIndex[key]))

    # Custom similarity score:
    # Numerator: Jaccard similarity
    # Denominator: Abs value of difference in length of ingredients + 1

    recipeScores = {}
    userIngredientSet = set(user_ingredients)
    for recipe in recipeSet:
        # grab ingredient set for recipe
        recipeIngredientSet = recipeToIngredients[recipe]
        # grab intersection b/w recipe ingredients and user ingredients
        intersection = len(recipeIngredientSet.intersection(userIngredientSet))
        # grab union b/w recipe ingredients and user ingredients
        union = len(recipeIngredientSet.union(userIngredientSet))
        # calc normalization factor
        normalizationFactor = abs(
            len(userIngredientSet) - len(recipeIngredientSet))
        # calc similarity score based on jaccard similarity equation
        similarityScore = (intersection) / \
            (abs(union * normalizationFactor) + 1)
        # store similarity score
        recipeScores[recipe] = similarityScore

    # sort recipe scores in descending order
    descendingScores = sorted(recipeScores.items(),
                              key=lambda x: x[1], reverse=True)

    # return top 10 recipes
    selected = 0
    finalList = []
    for key in descendingScores:
        # Returning tuple: recipe name, list of ingredients
        finalList.append((key[0], list(recipeToIngredients[key[0]])))
        selected += 1
        if (selected > 9):
            break

    return finalList


def searchRecipesCosine(user_ingredients):
    """Find top recipes based on cosine similarity to user ingredients list."""
    # map of ingredient to list of recipes that need that ingredient
    invertedIndex = {}
    # map of recipes to their list of ingredients
    recipeToIngredients = {}
    with open('dishTOingredients.output', "r") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if len(line):
                # build inverted index
                recipe = line.split(":")[0].strip()
                allIngredients = line.split(":")[1].strip()
                ingredientsList = allIngredients.split("|")
                for ingredient in ingredientsList:
                    if ingredient in invertedIndex:
                        invertedIndex[ingredient].append(recipe)
                    else:
                        invertedIndex[ingredient] = [recipe]
                    if recipe in recipeToIngredients:
                        recipeToIngredients[recipe].add(ingredient)
                    else:
                        recipeToIngredients[recipe] = set([ingredient])

            line = fp.readline()

    # set of all recipes that have at least one ingredient from the user inputted list of ingredients
    recipeSet = set()
    for ingredient in user_ingredients:
        for key in invertedIndex:
            if checkMatchingWords(ingredient, key):
                recipeSet = recipeSet.union(set(invertedIndex[key]))

    # Custom similarity score: cosine similarty
    # calculate cosine similarity using equation: np.dot(A, B) / (norm(A) * norm(B))
    # A = list of user ingredients
    # B = list of ingredients in recipe
    # calculate cosine similarity for all recipes, return top 10

    recipeScores = {}
    # for each relevant recipe, calculate cosine similarity to user ingredients
    for recipe in recipeSet:
        # store recipe ingredients as a list
        recipeIngredientSet = recipeToIngredients[recipe]
        # create tfidf vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        # build sparse matrix of tfidf values to convert text to float values
        sparse_matrix = tfidf_vectorizer.fit_transform(recipeIngredientSet)
        # create array from sparse matrix
        doc_term_matrix = sparse_matrix.toarray()

        # build matrix of tfidf values for user ingredients
        tgt_transform = tfidf_vectorizer.transform(user_ingredients)
        # calculate cosine similarity between recipe matrix and user input matrix
        tgt_cosine = cosine_similarity(doc_term_matrix, tgt_transform)
        # store cosine score for recipe
        recipeScores[recipe] = tgt_cosine

    # collapse np array of scores into single similarity measure value
    for recipe in recipeScores:
        recipeScores[recipe] = np.sum(recipeScores[recipe])

    # sort recipes by similarity score
    descendingScores = sorted(recipeScores.items(),
                              key=lambda x: x[1], reverse=True)

    # return top 10 results
    selected = 0
    finalList = []
    for key in descendingScores:
        # Returning tuple: recipe name, list of ingredients
        finalList.append((key[0], list(recipeToIngredients[key[0]])))
        selected += 1
        if (selected > 9):
            break

    return finalList


def searchRecipesLeven(user_ingredients):
    """Find top recipes based on levenshtein distance to user ingredients list."""
    # map of ingredient to list of recipes that need that ingredient
    invertedIndex = {}
    # map of recipes to their list of ingredients
    recipeToIngredients = {}
    with open('dishTOingredients.output', "r") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if len(line):
                # build inverted index
                recipe = line.split(":")[0].strip()
                allIngredients = line.split(":")[1].strip()
                ingredientsList = allIngredients.split("|")
                for ingredient in ingredientsList:
                    if ingredient in invertedIndex:
                        invertedIndex[ingredient].append(recipe)
                    else:
                        invertedIndex[ingredient] = [recipe]
                    if recipe in recipeToIngredients:
                        recipeToIngredients[recipe].add(ingredient)
                    else:
                        recipeToIngredients[recipe] = set([ingredient])

            line = fp.readline()

    # set of all recipes that have at least one ingredient from the user inputted list of ingredients
    recipeSet = set()
    for ingredient in user_ingredients:
        for key in invertedIndex:
            if checkMatchingWords(ingredient, key):
                recipeSet = recipeSet.union(set(invertedIndex[key]))

    # Custom similarity score: Levenshtein distance
    # find number of different ingredients between user ingredients and recipe ingredients
    # recipes with lower # differences = higher similarity score

    recipeScores = {}
    # for each relevant recipe calculate levenshtein distance to user ingredients
    for recipe in recipeSet:
        # store user and recipe ingredients as lists
        userTokens = flatten(user_ingredients)
        recipeTokens = flatten(list(recipeToIngredients[recipe]))
        # find levenshtein distance between user and recipe ingredients
        recipeScores[recipe] = levenshtein_distance(
            userTokens, recipeTokens)

    # sort recipes by similarity score (lower score == better match)
    ascendingScores = sorted(recipeScores.items(),
                             key=lambda x: x[1])

    # return top 10 results
    selected = 0
    finalList = []
    for key in ascendingScores:
        # Returning tuple: recipe name, list of ingredients
        finalList.append((key[0], list(recipeToIngredients[key[0]])))
        selected += 1
        if (selected > 9):
            break

    return finalList


def levenshtein_distance(list1, list2):
    """Compute the Levenshtein distance between two lists of words."""
    # create a matrix of zeros with dimensions (len(list1) + 1) x (len(list2) + 1)
    matrix = [[0] * (len(list2) + 1) for _ in range(len(list1) + 1)]

    # fill the first row and column of the matrix
    for i in range(len(list1) + 1):
        matrix[i][0] = i
    for j in range(len(list2) + 1):
        matrix[0][j] = j

    # loop through each word in list1 and list2, and compute the Levenshtein distance
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            # if words are same, levenstein distance is unchanged
            if list1[i-1] == list2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(
                    matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1]) + 1

    # return the Levenshtein distance score
    return matrix[-1][-1]


def reducedSimilarCosine(user_ingredients):
    """Find top recipes based on dimension reduction + cosine similarity to user ingredients list."""
    # map of ingredient to list of recipes that need that ingredient
    invertedIndex = {}
    # map of recipes to their list of ingredients
    recipeToIngredients = {}
    with open('dishTOingredients.output', "r") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if len(line):
                # build inverted index
                recipe = line.split(":")[0].strip()
                allIngredients = line.split(":")[1].strip()
                ingredientsList = allIngredients.split("|")
                for ingredient in ingredientsList:
                    if ingredient in invertedIndex:
                        invertedIndex[ingredient].append(recipe)
                    else:
                        invertedIndex[ingredient] = [recipe]
                    if recipe in recipeToIngredients:
                        recipeToIngredients[recipe].add(ingredient)
                    else:
                        recipeToIngredients[recipe] = set([ingredient])

            line = fp.readline()

    # set of all recipes that have at least one ingredient from the user inputted list of ingredients
    recipeSet = set()
    for ingredient in user_ingredients:
        for key in invertedIndex:
            if checkMatchingWords(ingredient, key):
                recipeSet = recipeSet.union(set(invertedIndex[key]))

    # Flatten the list of ingredients for all recipes into a single list of ingredients
    flat_list = []
    index_list = []
    for recipe in recipeSet:
        recipeIngredientSet = flatten(list(recipeToIngredients[recipe]))
        # store all ingredients found across all relevant recipes
        flat_list.append(recipeIngredientSet)
        index_list.append(recipe)

    # Convert list of lists to list of strings
    flat_list = [' '.join(recipe) for recipe in flat_list]

    # convert the list of strings into a sparse matrix
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(flat_list)

    # use UMAP to reduce the dimensionality of the sparse matrix
    umap = UMAP(n_components=2)
    reduced_data = umap.fit_transform(sparse_matrix)

    # convert the user's list of ingredients into a sparse matrix
    query_sparse = vectorizer.transform([' '.join(user_ingredients)])
    query_reduced = umap.transform(query_sparse)

    # calculate cosine similarities between the user's query and recipe vectors
    similarities = cosine_similarity(query_reduced, reduced_data)[0]

    # find the documents with the highest similarity scores
    most_similar_doc_indices = np.argsort(similarities)[::-1][:10]
    most_similar_docs = []
    for value in most_similar_doc_indices:
        most_similar_docs.append(index_list[value])

    # return top 10 results
    selected = 0
    finalList = []
    for key in most_similar_docs:
        # Returning tuple: recipe name, list of ingredients
        finalList.append((key, list(recipeToIngredients[key])))
        selected += 1
        if (selected > 9):
            break

    return finalList


def flatten(ingredientList):
    """Flatten list of ingredients."""
    resultList = []

    # for each ingredient, split into list of words if multi-word
    for str in ingredientList:
        words = str.split()
        resultList.extend(words)

    # create final list of all words in ingredient list
    flattened_list = list(itertools.chain.from_iterable(
        [item.split() for item in resultList]))

    return flattened_list
