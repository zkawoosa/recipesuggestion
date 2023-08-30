from bs4 import BeautifulSoup
import requests
import concurrent.futures

dish_to_ingredients = {}
ingredient_to_dishes = {}
counter = 0

def request_function(recipe_link, session):
    # For a given link, retrieve page
    with session.get(recipe_link) as response:
        # Create beautiful soup object, extract ingredients list for one recipe
        parser = BeautifulSoup(response.content, "html.parser")
        recipe_name = parser.find(id="article-heading_1-0")
        recipe_ingredient_list = parser.find(
            class_="mntl-structured-ingredients__list")

        # For valid recipe, append to output files (mapped in both directions)
        if recipe_name is not None:
            recipe_name = str(recipe_name.text)
            for ingredient in recipe_ingredient_list.children:
                if ingredient != '\n' and not None:
                    # quantity = ingredient.find('span', {'data-ingredient-quantity': True})
                    # units = ingredient.find('span', {'data-ingredient-unit': True})
                    ingredient_name = ingredient.find(
                        'span', {'data-ingredient-name': True})
                    ingredient_name = str(ingredient_name.text)

                    if recipe_name not in dish_to_ingredients:
                        dish_to_ingredients[recipe_name] = [ingredient_name]
                    else:
                        dish_to_ingredients[recipe_name].append(ingredient_name)

                    if ingredient_name not in ingredient_to_dishes:
                        ingredient_to_dishes[ingredient_name] = [recipe_name[1:]]
                    else:
                        ingredient_to_dishes[ingredient_name].append(recipe_name[1:])

                    # print("Quantity: ", quantity.text)
                    # print("Units: ", units.text)

            print("Name: ", recipe_name)
            print("Link: ", recipe_link)

        # print("Link: ", recipe_link)
        # print("Header: ", recipe_name)
        # print("Ingredients: ", recipe_ingredient_list)
        # print("extractor.py")

def main():
    with open('crawler.output', 'r') as f:
        recipe_links = f.read().splitlines()

    # Use same connection for each request for efficiency purposes
    get_session = requests.Session()

    # Implement multithreading to run 10 requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit requests to the executor and store the futures in a list
        futures = [executor.submit(request_function, url, get_session) for url in recipe_links]
        # Wait for all the futures to complete and get their results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    print(ingredient_to_dishes)
    # Output recipe names mapped to ingredient lists
    with open('dishTOingredients.output', 'w') as f:
        for key, value in dish_to_ingredients.items():
            f.write(key + ': ' + ' | '.join(value))
    # Output ingredient lists mapped to recipe names
    with open('ingredientTOdishes.output', 'w') as f:
        for key, value in ingredient_to_dishes.items():
            f.write(key + ': ' + ' | '.join(value) + '\n')


if __name__ == '__main__':
    main()
