from flask import Flask, render_template, request
from match import searchRecipes, searchRecipesCosine, searchRecipesLeven, reducedSimilarCosine, flatten

app = Flask(__name__)


@app.route('/')
def home():
    """Display the home page."""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """Perform search if user ingredients submitted."""
    # grab user ingredients from text box
    input = request.form['search-box']
    # split ingredients into list
    input = input.split(',')
    # search for most similar recipes based on user input
    recipes = searchRecipes(input)
    # find matching ingredients b/w recipe and user input
    matches = findSimilarIngredients(recipes, input)
    # create context dictionary for html rendering
    context = {"results": recipes, "matches": matches}
    # calculate macro precision and recall across all returned recipes
    calculate_metrics(matches)
    # render context dictionary in html to display recipe results
    return render_template('index.html', **context)


def findSimilarIngredients(recipes, input):
    """Find similar ingredients to the ones in the recipes."""
    matches = {}
    # loop through each recipe and find matching ingredients
    for recipe in recipes:
        # store set of tokens in recipe's ingredients
        recipeTokensSet = set(flatten(recipe[1]))
        # store set of tokens in users ingredients
        userTokensSet = set(flatten(input))
        # find matching ingredients
        matches[recipe[0]] = recipeTokensSet & userTokensSet
        # calculate precision + recall based on # of matches
        precision = len(matches[recipe[0]]) / len(recipeTokensSet)
        recall = len(matches[recipe[0]]) / len(userTokensSet)
        # store precision and recall in matches dictionary
        matches[recipe[0]] = (matches[recipe[0]], precision, recall)
    # return dictionary of matching ingredients for each recipe
    return matches


def calculate_metrics(matches):
    """Calculate macro precision and recall across all returned recipes."""
    macroP = 0.0
    macroR = 0.0
    f1s = {}
    count = 1
    # loop through each recipe and calculate macro precision and recall
    for item in matches:
        macroP += matches[item][1]
        macroR += matches[item][2]
        # calculate f1 score at each rank (1-10)
        f1s[count] = 2 * ((matches[item][1] / count) * (matches[item][2] / count)) / \
            (((matches[item][1] / count) + 1) +
             ((matches[item][2] / count) + 1))
        count += 1

    # calculate macro precision and recall
    macroP /= len(matches)
    macroR /= len(matches)

    # calculate final f1 score
    f1_final = 2 * (macroP * macroR) / ((macroP + 1) + (macroR + 1))

    # print results
    print("Macro Precision: ", macroP)
    print("Macro Recall: ", macroR)
    print("F1 Score Total: ", f1_final)
    for f1 in f1s:
        print("F1 Score at Rank {}: {}".format(f1, f1s[f1]))


if __name__ == '__main__':
    app.run(debug=True)
