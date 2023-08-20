import random
import spacy
from spacy.training import Example
TRAINING_DATA = [{'text': 'I need a casual cotton dress for a summer party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 42, 'end': 47, 'value': 'party', 'entity': 'occasion'}, {'start': 16, 'end': 22, 'value': 'cotton', 'entity': 'material'}, {'start': 23, 'end': 28, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 35, 'end': 41, 'value': 'summer', 'entity': 'weather'}]},
        {'text': 'Looking for a fancy frock for a wedding.', 'intent': 'get_attributes', 'entities': [{'start': 14, 'end': 19, 'value': 'fancy', 'entity': 'style'}, {'start': 32, 'end': 39, 'value': 'wedding', 'entity': 'occasion'}, {'start': 20, 'end': 25, 'value': 'frock', 'entity': 'outfit_type'}]},
        {'text': 'I want a plain chiffon saree for a formal event.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'plain', 'entity': 'style'}, {'start': 35, 'end': 47, 'value': 'formal', 'entity': 'occasion'}, {'start': 15, 'end': 22, 'value': 'chiffon', 'entity': 'material'}, {'start': 23, 'end': 28, 'value': 'saree', 'entity': 'outfit_type'}]},
        {'text': "I'm searching for a casual shirt for beach vacation.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 26, 'value': 'casual', 'entity': 'style'}, {'start': 45, 'end': 53, 'value': 'beach vacation', 'entity': 'occasion'}, {'start': 27, 'end': 32, 'value': 'shirt', 'entity': 'outfit_type'}]},
        {'text': 'Looking for leather boots for a rainy day.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 19, 'value': 'leather', 'entity': 'material'}, {'start': 32, 'end': 37, 'value': 'rainy', 'entity': 'weather'}, {'start': 20, 'end': 25, 'value': 'boots', 'entity': 'shoe_type'}]},
        {'text': 'I need a georgette jacket for a party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 18, 'value': 'georgette', 'entity': 'material'}, {'start': 32, 'end': 37, 'value': 'party', 'entity': 'style'}, {'start': 19, 'end': 25, 'value': 'jacket', 'entity': 'outfit_type'}]},
        {'text': 'I want a formal saree for a wedding ceremony.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'formal', 'entity': 'style'}, {'start': 28, 'end': 35, 'value': 'wedding', 'entity': 'occasion'}, {'start': 16, 'end': 21, 'value': 'saree', 'entity': 'outfit_type'}]},
        {'text': 'Looking for a party top with fancy embroidery.', 'intent': 'get_attributes', 'entities': [{'start': 14, 'end': 19, 'value': 'party', 'entity': 'style'}, {'start': 29, 'end': 34, 'value': 'fancy', 'entity': 'style'}, {'start': 35, 'end': 45, 'value': 'embroidery', 'entity': 'style'}, {'start': 20, 'end': 23, 'value': 'top', 'entity': 'outfit_type'}]},
        {'text': "I need a casual frock for a kids' birthday party.", 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 43, 'end': 48, 'value': "party", 'entity': 'occasion'}, {'start': 16, 'end': 21, 'value': 'frock', 'entity': 'outfit_type'}]},
        {'text': 'I want a sleeveless georgette dress for a summer vacation.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 19, 'value': 'sleeveless', 'entity': 'sleeve'}, {'start': 20, 'end': 29, 'value': 'georgette', 'entity': 'material'}, {'start': 49, 'end': 57, 'value': 'vacation', 'entity': 'occasion'}, {'start': 30, 'end': 35, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 42, 'end': 48, 'value': 'summer', 'entity': 'weather'}]},
        {'text': 'Looking for a formal shirt for a business meeting.', 'intent': 'get_attributes', 'entities': [{'start': 14, 'end': 20, 'value': 'formal', 'entity': 'style'}, {'start': 42, 'end': 49, 'value': 'meeting', 'entity': 'occasion'}, {'start': 21, 'end': 26, 'value': 'shirt', 'entity': 'outfit_type'}]},
        {'text': 'I need a casual green dress for a summer party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'green', 'entity': 'color'}, {'start': 41, 'end': 46, 'value': 'party', 'entity': 'occasion'}, {'start': 22, 'end': 27, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 34, 'end': 40, 'value': 'summer', 'entity': 'weather'}]},
        {'text': "I'm searching for a formal black saree for a wedding.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 26, 'value': 'formal', 'entity': 'style'}, {'start': 27, 'end': 32, 'value': 'black', 'entity': 'color'}, {'start': 45, 'end': 52, 'value': 'wedding', 'entity': 'occasion'}, {'start': 33, 'end': 38, 'value': 'saree', 'entity': 'outfit_type'}]},
        {'text': 'I want a plain red chiffon saree for a formal event.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'plain', 'entity': 'style'}, {'start': 15, 'end': 18, 'value': 'red', 'entity': 'color'}, {'start': 39, 'end': 45, 'value': 'formal', 'entity': 'occasion'}, {'start': 19, 'end': 26, 'value': 'chiffon', 'entity': 'material'}, {'start': 27, 'end': 32, 'value': 'saree', 'entity': 'outfit_type'}]},
        {'text': "I need a casual blue frock for a kids' birthday party.", 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 16, 'end': 20, 'value': 'blue', 'entity': 'color'}, {'start': 48, 'end': 53, 'value': "party", 'entity': 'occasion'}, {'start': 21, 'end': 26, 'value': 'frock', 'entity': 'outfit_type'}]},
        {'text': 'Looking for leather brown boots for a rainy day.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 19, 'value': 'leather', 'entity': 'material'}, {'start': 20, 'end': 25, 'value': 'brown', 'entity': 'color'}, {'start': 38, 'end': 43, 'value': 'rainy', 'entity': 'weather'}, {'start': 26, 'end': 31, 'value': 'boots', 'entity': 'shoe_type'}]},
        {'text': 'I need a georgette pink jacket for a party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 18, 'value': 'georgette', 'entity': 'material'}, {'start': 19, 'end': 23, 'value': 'pink', 'entity': 'color'}, {'start': 37, 'end': 42, 'value': 'party', 'entity': 'style'}, {'start': 24, 'end': 30, 'value': 'jacket', 'entity': 'outfit_type'}]},
        {'text': 'Looking for a floral cotton top for a casual outing.', 'intent': 'get_attributes', 'entities': [{'start': 14, 'end': 20, 'value': 'floral', 'entity': 'style'}, {'start': 21, 'end': 27, 'value': 'cotton', 'entity': 'material'}, {'start': 38, 'end': 44, 'value': 'casual', 'entity': 'occasion'}, {'start': 28, 'end': 31, 'value': 'top', 'entity': 'outfit_type'}]},
        {'text': "I'm looking for a striped satin blouse for a formal event.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 25, 'value': 'striped', 'entity': 'style'}, {'start': 26, 'end': 31, 'value': 'satin', 'entity': 'material'}, {'start': 45, 'end': 51, 'value': 'formal', 'entity': 'occasion'}, {'start': 32, 'end': 38, 'value': 'blouse', 'entity': 'outfit_type'}]},
        {'text': 'Looking for open toe heels in synthetic material.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 20, 'value': 'open toe', 'entity': 'shoe_type'}, {'start': 21, 'end': 26, 'value': 'heels', 'entity': 'shoe_type'}, {'start': 30, 'end': 39, 'value': 'synthetic', 'entity': 'material'}]},
        {'text': 'I need a yellow chiffon dress for a summer beach vacation.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'yellow', 'entity': 'color'}, {'start': 16, 'end': 23, 'value': 'chiffon', 'entity': 'material'}, {'start': 49, 'end': 57, 'value': 'vacation', 'entity': 'occasion'}, {'start': 24, 'end': 29, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 36, 'end': 42, 'value': 'summer', 'entity': 'weather'}]},
        {'text': 'I need a formal black suit for a business conference.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'formal', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'black', 'entity': 'color'}, {'start': 22, 'end': 26, 'value': 'suit', 'entity': 'outfit_type'}, {'start': 33, 'end': 41, 'value': 'business', 'entity': 'occasion'}]},
        {'text': 'I want a traditional red kimono for a Japanese festival.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 20, 'value': 'traditional', 'entity': 'style'}, {'start': 21, 'end': 24, 'value': 'red', 'entity': 'color'}, {'start': 25, 'end': 31, 'value': 'kimono', 'entity': 'outfit_type'}, {'start': 47, 'end': 55, 'value': 'festival', 'entity': 'occasion'}]},
        {'text': 'I need a casual denim jacket for a weekend outing.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'denim', 'entity': 'material'}, {'start': 22, 'end': 28, 'value': 'jacket', 'entity': 'outfit_type'}, {'start': 43, 'end': 49, 'value': 'outing', 'entity': 'occasion'}]},
        {'text': "I'm searching for a beachy white maxi dress for a vacation.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 26, 'value': 'beachy', 'entity': 'style'}, {'start': 27, 'end': 32, 'value': 'white', 'entity': 'color'}, {'start': 33, 'end': 37, 'value': 'maxi', 'entity': 'outfit_type'}, {'start': 50, 'end': 58, 'value': 'vacation', 'entity': 'occasion'}]},
        {'text': 'I need a sporty tracksuit for my gym sessions.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'sporty', 'entity': 'style'}, {'start': 16, 'end': 25, 'value': 'tracksuit', 'entity': 'outfit_type'}, {'start': 33, 'end': 36, 'value': 'gym', 'entity': 'occasion'}]},
        {'text': "I'm looking for a cozy sweater for the winter season.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 22, 'value': 'cozy', 'entity': 'style'}, {'start': 23, 'end': 30, 'value': 'sweater', 'entity': 'outfit_type'}, {'start': 39, 'end': 45, 'value': 'winter', 'entity': 'weather'}]},
        {'text': 'I need a traditional African print dress for a cultural event.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 20, 'value': 'traditional', 'entity': 'style'}, {'start': 21, 'end': 28, 'value': 'African', 'entity': 'region'}, {'start': 29, 'end': 34, 'value': 'print', 'entity': 'style'}, {'start': 35, 'end': 40, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 47, 'end': 55, 'value': 'cultural', 'entity': 'occasion'}]},
        {'text': "I'm searching for a stylish leather jacket for a night out.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 27, 'value': 'stylish', 'entity': 'style'}, {'start': 28, 'end': 35, 'value': 'leather', 'entity': 'material'}, {'start': 36, 'end': 42, 'value': 'jacket', 'entity': 'outfit_type'}, {'start': 49, 'end': 58, 'value': 'night out', 'entity': 'occasion'}]},
        {'text': 'I need a summer dress with floral print for a garden party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'summer', 'entity': 'weather'}, {'start': 16, 'end': 21, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 27, 'end': 33, 'value': 'floral', 'entity': 'style'}, {'start': 53, 'end': 58, 'value': 'party', 'entity': 'occasion'}]},
        {'text': 'I want a cozy woolen sweater for the chilly weather.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 13, 'value': 'cozy', 'entity': 'style'}, {'start': 14, 'end': 20, 'value': 'woolen', 'entity': 'material'}, {'start': 21, 'end': 28, 'value': 'sweater', 'entity': 'outfit_type'}, {'start': 37, 'end': 51, 'value': 'chilly weather', 'entity': 'weather'}]},
        {'text': "I'm looking for a traditional hanbok for a Korean festival.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 29, 'value': 'traditional', 'entity': 'style'}, {'start': 30, 'end': 36, 'value': 'hanbok', 'entity': 'outfit_type'}, {'start': 43, 'end': 49, 'value': 'Korean', 'entity': 'region'}, {'start': 50, 'end': 58, 'value': 'festival', 'entity': 'occasion'}]},
        {'text': 'I need a formal navy blue suit for a wedding ceremony.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'formal', 'entity': 'style'}, {'start': 16, 'end': 25, 'value': 'navy blue', 'entity': 'color'}, {'start': 26, 'end': 30, 'value': 'suit', 'entity': 'outfit_type'}, {'start': 37, 'end': 44, 'value': 'wedding', 'entity': 'occasion'}]},
        {'text': "I'm searching for a casual plaid shirt for a picnic.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 26, 'value': 'casual', 'entity': 'style'}, {'start': 27, 'end': 32, 'value': 'plaid', 'entity': 'pattern'}, {'start': 33, 'end': 38, 'value': 'shirt', 'entity': 'outfit_type'}, {'start': 45, 'end': 51, 'value': 'picnic', 'entity': 'occasion'}]},
        {'text': 'I need a cozy knitted sweater for a fireside gathering.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 13, 'value': 'cozy', 'entity': 'style'}, {'start': 14, 'end': 21, 'value': 'knitted', 'entity': 'material'}, {'start': 22, 'end': 29, 'value': 'sweater', 'entity': 'outfit_type'}, {'start': 36, 'end': 54, 'value': 'fireside gathering', 'entity': 'occasion'}]},
        {'text': "I'm searching for a flowy floral dress for a garden party.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 25, 'value': 'flowy', 'entity': 'style'}, {'start': 26, 'end': 32, 'value': 'floral', 'entity': 'style'}, {'start': 33, 'end': 38, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 52, 'end': 57, 'value': 'party', 'entity': 'occasion'}]},
        {'text': 'I want a sleek black cocktail dress for a night out.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'sleek', 'entity': 'style'}, {'start': 15, 'end': 20, 'value': 'black', 'entity': 'color'}, {'start': 21, 'end': 29, 'value': 'cocktail', 'entity': 'style'}, {'start': 42, 'end': 51, 'value': 'night out', 'entity': 'occasion'}]},
        {'text': 'I need a bohemian printed dress for a music festival.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 17, 'value': 'bohemian', 'entity': 'style'}, {'start': 18, 'end': 25, 'value': 'printed', 'entity': 'style'}, {'start': 26, 'end': 31, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 44, 'end': 52, 'value': 'festival', 'entity': 'occasion'}]},
        {'text': 'I want a floral frock with pink and white colors.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'floral', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'frock', 'entity': 'top_type'}, {'start': 27, 'end': 31, 'value': 'pink', 'entity': 'color'}, {'start': 36, 'end': 41, 'value': 'white', 'entity': 'color'}]},
        {'text': 'I need a casual jeans and a t shirt with a blue color.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'jeans', 'entity': 'bottom_type'}, {'start': 28, 'end': 35, 'value': 't shirt', 'entity': 'outfit_type'}, {'start': 43, 'end': 47, 'value': 'blue', 'entity': 'color'}]},
        {'text': "I'm looking for a traditional salwar kameez with vibrant colors.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 29, 'value': 'traditional', 'entity': 'style'}, {'start': 30, 'end': 43, 'value': 'salwar kameez', 'entity': 'outfit_type'}, {'start': 49, 'end': 56, 'value': 'vibrant', 'entity': 'style'}, {'start': 57, 'end': 63, 'value': 'colors', 'entity': 'outfit_type'}]},
        {'text': 'I want a black kurta and leggings for a casual outing.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'black', 'entity': 'color'}, {'start': 15, 'end': 20, 'value': 'kurta', 'entity': 'outfit_type'}, {'start': 25, 'end': 33, 'value': 'leggings', 'entity': 'bottom_type'}, {'start': 40, 'end': 46, 'value': 'casual', 'entity': 'style'}, {'start': 47, 'end': 53, 'value': 'outing', 'entity': 'occasion'}]},
        {'text': 'I need a stylish jeans and a trendy top with vibrant colors.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 16, 'value': 'stylish', 'entity': 'style'}, {'start': 17, 'end': 22, 'value': 'jeans', 'entity': 'bottom_type'}, {'start': 29, 'end': 35, 'value': 'trendy', 'entity': 'style'}, {'start': 36, 'end': 39, 'value': 'top', 'entity': 'outfit_type'}, {'start': 45, 'end': 52, 'value': 'vibrant', 'entity': 'style'}, {'start': 53, 'end': 59, 'value': 'colors', 'entity': 'outfit_type'}]},
        {'text': "I'm searching for a comfortable leggings and a plain t shirt.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 31, 'value': 'comfortable', 'entity': 'style'}, {'start': 32, 'end': 40, 'value': 'leggings', 'entity': 'bottom_type'}, {'start': 47, 'end': 52, 'value': 'plain', 'entity': 'style'}, {'start': 53, 'end': 60, 'value': 't shirt', 'entity': 'outfit_type'}]},
        {'text': 'I want a classic jeans and a white shirt for a formal event.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 16, 'value': 'classic', 'entity': 'style'}, {'start': 17, 'end': 22, 'value': 'jeans', 'entity': 'bottom_type'}, {'start': 29, 'end': 34, 'value': 'white', 'entity': 'color'}, {'start': 35, 'end': 40, 'value': 'shirt', 'entity': 'outfit_type'}, {'start': 47, 'end': 53, 'value': 'formal', 'entity': 'style'}, {'start': 54, 'end': 59, 'value': 'event', 'entity': 'occasion'}]},
        {'text': 'I need a trendy top and jeans for a night out.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'trendy', 'entity': 'style'}, {'start': 16, 'end': 19, 'value': 'top', 'entity': 'outfit_type'}, {'start': 24, 'end': 29, 'value': 'jeans', 'entity': 'outfit_type'}, {'start': 36, 'end': 45, 'value': 'night out', 'entity': 'occasion'}]},
        {'text': "I'm looking for a stylish kurta and leggings with vibrant colors.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 25, 'value': 'stylish', 'entity': 'style'}, {'start': 26, 'end': 31, 'value': 'kurta', 'entity': 'outfit_type'}, {'start': 36, 'end': 44, 'value': 'leggings', 'entity': 'bottom_type'}, {'start': 50, 'end': 57, 'value': 'vibrant', 'entity': 'style'}, {'start': 58, 'end': 64, 'value': 'colors', 'entity': 'outfit_type'}]},
        {'text': 'I want a casual jeans and a plain t shirt for a relaxing day.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'jeans', 'entity': 'bottom_type'}, {'start': 28, 'end': 33, 'value': 'plain', 'entity': 'style'}, {'start': 34, 'end': 41, 'value': 't shirt', 'entity': 'outfit_type'}, {'start': 48, 'end': 60, 'value': 'relaxing day', 'entity': 'occasion'}]},
        {'text': 'I need a comfy leggings and a cozy sweater for a chilly day.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'comfy', 'entity': 'style'}, {'start': 15, 'end': 23, 'value': 'leggings', 'entity': 'bottom_type'}, {'start': 30, 'end': 34, 'value': 'cozy', 'entity': 'style'}, {'start': 35, 'end': 42, 'value': 'sweater', 'entity': 'outfit_type'}, {'start': 49, 'end': 59, 'value': 'chilly day', 'entity': 'occasion'}]},
        {'text': 'I want a trendy skirt and a floral top with vibrant colors.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'trendy', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'skirt', 'entity': 'bottom_type'}, {'start': 28, 'end': 34, 'value': 'floral', 'entity': 'style'}, {'start': 35, 'end': 38, 'value': 'top', 'entity': 'outfit_type'}, {'start': 44, 'end': 51, 'value': 'vibrant', 'entity': 'style'}, {'start': 52, 'end': 58, 'value': 'colors', 'entity': 'outfit_type'}]},
        {'text': "I'm searching for a classic kurta and leggings with a white color.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 27, 'value': 'classic', 'entity': 'style'}, {'start': 28, 'end': 33, 'value': 'kurta', 'entity': 'outfit_type'}, {'start': 38, 'end': 46, 'value': 'leggings', 'entity': 'bottom_type'}, {'start': 54, 'end': 59, 'value': 'white', 'entity': 'color'}]},
        {'text': 'Can you suggest some ankle strap shoes for a party?', 'intent': 'get_attributes', 'entities': [{'start': 21, 'end': 32, 'value': 'ankle strap', 'entity': 'shoe_type'}, {'start': 45, 'end': 50, 'value': 'party', 'entity': 'occasion'}]},
        {'text': 'I need open toe shoes to go with my summer outfit.', 'intent': 'get_attributes', 'entities': [{'start': 7, 'end': 15, 'value': 'open toe', 'entity': 'shoe_type'}, {'start': 36, 'end': 42, 'value': 'summer', 'entity': 'weather'}, {'start': 43, 'end': 49, 'value': 'outfit', 'entity': 'outfit_type'}]},
        {'text': 'Please recommend some falt shoes for everyday wear.', 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'flat', 'entity': 'shoe_type'}, {'start': 37, 'end': 45, 'value': 'everyday', 'entity': 'occasion'}]},
        {'text': 'What type of bag should I carry with my casual outfit?', 'intent': 'get_attributes', 'entities': [{'start': 40, 'end': 46, 'value': 'casual', 'entity': 'style'}, {'start': 47, 'end': 53, 'value': 'outfit', 'entity': 'top_type'}]},
        {'text': 'Can you suggest some heals shoes for a special occasion?', 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 4, 'value': 'heels', 'entity': 'shoe_type'}, {'start': 39, 'end': 46, 'value': 'special', 'entity': 'occasion'}]},
        {'text': "I'm looking for a shoulder bag to match my casual outfit.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 30, 'value': 'shoulder bag', 'entity': 'bag_type'}, {'start': 43, 'end': 49, 'value': 'casual', 'entity': 'style'}, {'start': 50, 'end': 56, 'value': 'outfit', 'entity': 'outfit_type'}]},
        {'text': 'I need a clutch bag for the wedding ceremony.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'clutch', 'entity': 'bag_type'}, {'start': 28, 'end': 35, 'value': 'wedding', 'entity': 'occasion'}]},
        {'text': 'What type of bottom should I wear with a floral top?', 'intent': 'get_attributes', 'entities': [{'start': 41, 'end': 47, 'value': 'floral', 'entity': 'style'}, {'start': 48, 'end': 51, 'value': 'top', 'entity': 'outfit_type'}]},
        {'text': "I'm searching for a fancy handbag for a fancy event.", 'intent': 'get_attributes', 'entities': [{'start': 26, 'end': 33, 'value': 'handbag', 'entity': 'bag_type'}, {'start': 20, 'end': 25, 'value': 'fancy', 'entity': 'style'}, {'start': 46, 'end': 51, 'value': 'event', 'entity': 'occasion'}]},
        {'text': 'Can you recommend a leather bag for my formal outfit?', 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 27, 'value': 'leather', 'entity': 'material'}, {'start': 39, 'end': 45, 'value': 'formal', 'entity': 'style'}, {'start': 46, 'end': 52, 'value': 'outfit', 'entity': 'outfit_type'}]},
        {'text': 'I need a pair of sandals for my beach vacation.', 'intent': 'get_attributes', 'entities': [{'start': 17, 'end': 24, 'value': 'sandals', 'entity': 'shoe_type'}, {'start': 38, 'end': 46, 'value': 'vacation', 'entity': 'occasion'}]},
        {'text': 'Suggest some jeans that go well with a striped shirt.', 'intent': 'get_attributes', 'entities': [{'start': 13, 'end': 18, 'value': 'jeans', 'entity': 'bottom_type'}, {'start': 39, 'end': 46, 'value': 'striped', 'entity': 'style'}, {'start': 47, 'end': 52, 'value': 'shirt', 'entity': 'outfit_type'}]},
        {'text': "I'm looking for a fancy skirt and a plain top.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 23, 'value': 'fancy', 'entity': 'style'}, {'start': 24, 'end': 29, 'value': 'skirt', 'entity': 'bottom_type'}, {'start': 36, 'end': 41, 'value': 'plain', 'entity': 'style'}, {'start': 42, 'end': 45, 'value': 'top', 'entity': 'outfit_type'}]},
        {'text': 'Can you suggest a top to match my jeans?', 'intent': 'get_attributes', 'entities': [{'start': 34, 'end': 39, 'value': 'jeans', 'entity': 'bottom_type'}, {'start': 18, 'end': 21, 'value': 'top', 'entity': 'outfit_type'}]},
        {'text': 'What kind of shoe goes well with a floral dress?', 'intent': 'get_attributes', 'entities': [{'start': 35, 'end': 41, 'value': 'floral', 'entity': 'style'}, {'start': 42, 'end': 47, 'value': 'dress', 'entity': 'outfit_type'}]},
        {'text': 'I need a formal shirt to wear with my trousers.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'formal', 'entity': 'style'}, {'start': 16, 'end': 21, 'value': 'shirt', 'entity': 'outfit_type'}, {'start': 38, 'end': 46, 'value': 'trousers', 'entity': 'bottom_type'}]},
        {'text': 'What type of bag should I carry with my frock?', 'intent': 'get_attributes', 'entities': [{'start': 40, 'end': 45, 'value': 'frock', 'entity': 'outfit_type'}]},
        {'text': 'I need a handbag to match my formal attire.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 16, 'value': 'handbag', 'entity': 'bag_type'}, {'start': 29, 'end': 35, 'value': 'formal', 'entity': 'style'}, {'start': 36, 'end': 42, 'value': 'attire', 'entity': 'outfit_type'}]},
        {'text': 'Suggest a bottom to go with my plain t shirt.', 'intent': 'get_attributes', 'entities': [{'start': 31, 'end': 44, 'value': 'plain t shirt', 'entity': 'outfit_type'}, {'start': 10, 'end': 16, 'value': 'bottom', 'entity': 'outfit_type'}]},
        {'text': 'What type of shoe should I wear with my casual outfit?', 'intent': 'get_attributes', 'entities': [{'start': 40, 'end': 46, 'value': 'casual', 'entity': 'style'}, {'start': 47, 'end': 53, 'value': 'outfit', 'entity': 'outfit_type'}]},
        {'text': 'I need a bag to match my summer dress.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 12, 'value': 'bag', 'entity': 'outfit_type'}, {'start': 25, 'end': 31, 'value': 'summer', 'entity': 'weather'}, {'start': 32, 'end': 37, 'value': 'dress', 'entity': 'outfit_type'}]},
        {'text': 'Suggest a handbag for my fancy event.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 17, 'value': 'handbag', 'entity': 'bag_type'}, {'start': 25, 'end': 30, 'value': 'fancy', 'entity': 'style'}, {'start': 31, 'end': 36, 'value': 'event', 'entity': 'occasion'}]},
        {'text': 'I need a comfortable pair of flats for everyday wear.', 'intent': 'get_attributes', 'entities': [{'start': 29, 'end': 34, 'value': 'flats', 'entity': 'shoe_type'}, {'start': 39, 'end': 47, 'value': 'everyday', 'entity': 'occasion'}]},
        {'text': 'Can you suggest a bag for my formal office outfit?', 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 21, 'value': 'bag', 'entity': 'outfit_type'}, {'start': 29, 'end': 35, 'value': 'formal', 'entity': 'style'}, {'start': 36, 'end': 42, 'value': 'office', 'entity': 'occasion'}, {'start': 43, 'end': 49, 'value': 'outfit', 'entity': 'outfit_type'}]},
        {'text': 'What type of bottom should I wear with my frock?', 'intent': 'get_attributes', 'entities': [{'start': 42, 'end': 47, 'value': 'frock', 'entity': 'outfit_type'}]},
        {'text': "I'm looking for a clutch bag for a night party.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 28, 'value': 'clutch bag', 'entity': 'bag_type'}, {'start': 41, 'end': 46, 'value': 'party', 'entity': 'occasion'}]},
        {'text': 'I need an outfit for my 5 year old daughter.', 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 34, 'value': '5 year old', 'entity': 'age'}, {'start': 35, 'end': 43, 'value': 'daughter', 'entity': 'gender'}]},
        {'text': 'Recommend a dress for a 7 year old girl.', 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 34, 'value': '7 year old', 'entity': 'age'}, {'start': 35, 'end': 39, 'value': 'girl', 'entity': 'gender'}]},
        {'text': "What's a suitable outfit for an 8 year old boy?", 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 42, 'value': '8 year old', 'entity': 'age'}, {'start': 43, 'end': 46, 'value': 'boy', 'entity': 'gender'}]},
        {'text': "I'm looking for clothes for a 6 year old.", 'intent': 'get_attributes', 'entities': [{'start': 30, 'end': 40, 'value': '6 year old', 'entity': 'age'}]},
        {'text': 'Suggest a summer outfit for my 9 year old son.', 'intent': 'get_attributes', 'entities': [{'start': 31, 'end': 41, 'value': '9 year old', 'entity': 'age'}, {'start': 42, 'end': 45, 'value': 'son', 'entity': 'gender'}, {'start': 10, 'end': 16, 'value': 'summer', 'entity': 'weather'}]},
        {'text': "What's a good outfit for a 4 year old boy?", 'intent': 'get_attributes', 'entities': [{'start': 27, 'end': 37, 'value': '4 year old', 'entity': 'age'}, {'start': 38, 'end': 41, 'value': 'boy', 'entity': 'gender'}]},
        {'text': 'I need a formal dress for my 10 year old daughter.', 'intent': 'get_attributes', 'entities': [{'start': 29, 'end': 40, 'value': '10 year old', 'entity': 'age'}, {'start': 41, 'end': 49, 'value': 'daughter', 'entity': 'gender'}, {'start': 9, 'end': 15, 'value': 'formal', 'entity': 'occasion'}]},
        {'text': 'Recommend an outfit for an 8 year old girl.', 'intent': 'get_attributes', 'entities': [{'start': 27, 'end': 37, 'value': '8 year old', 'entity': 'age'}, {'start': 38, 'end': 42, 'value': 'girl', 'entity': 'gender'}]},
        {'text': "What's a trendy outfit for a 7 year old?", 'intent': 'get_attributes', 'entities': [{'start': 29, 'end': 39, 'value': '7 year old', 'entity': 'age'}]},
        {'text': "I'm looking for clothes for a 5 year old boy.", 'intent': 'get_attributes', 'entities': [{'start': 30, 'end': 40, 'value': '5 year old', 'entity': 'age'}, {'start': 41, 'end': 44, 'value': 'boy', 'entity': 'gender'}]},
        {'text': 'Suggest a casual outfit for a young man.', 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 10, 'end': 16, 'value': 'casual', 'entity': 'occasion'}]},
        {'text': "What's a stylish outfit for a man in his 30s?", 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 41, 'end': 44, 'value': '30s', 'entity': 'age'}, {'start': 9, 'end': 16, 'value': 'stylish', 'entity': 'style'}]},
        {'text': 'Recommend a formal outfit for an elderly man.', 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 33, 'end': 40, 'value': 'elderly', 'entity': 'age'}, {'start': 12, 'end': 18, 'value': 'formal', 'entity': 'occasion'}]},
        {'text': "I'm looking for a trendy outfit for a young guy.", 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 38, 'end': 43, 'value': 'young', 'entity': 'age'}, {'start': 18, 'end': 24, 'value': 'trendy', 'entity': 'style'}]},
        {'text': 'Suggest a comfortable outfit for a middle aged man.', 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 35, 'end': 46, 'value': 'middle aged', 'entity': 'age'}, {'start': 10, 'end': 21, 'value': 'comfortable', 'entity': 'style'}]},
        {'text': "What's a suitable outfit for a man in his 40s?", 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 42, 'end': 45, 'value': '40s', 'entity': 'age'}]},
        {'text': 'Recommend a formal attire for a gentleman.', 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 41, 'value': 'gentleman', 'entity': 'gender'}, {'start': 12, 'end': 18, 'value': 'formal', 'entity': 'style'}]},
        {'text': 'I need a classic outfit for a man in his 50s.', 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 41, 'end': 44, 'value': '50s', 'entity': 'age'}, {'start': 9, 'end': 16, 'value': 'classic', 'entity': 'style'}]},
        {'text': 'Suggest a casual outfit for a young gentleman.', 'intent': 'get_attributes', 'entities': [{'start': 36, 'end': 45, 'value': 'gentleman', 'entity': 'gender'}, {'start': 10, 'end': 16, 'value': 'casual', 'entity': 'occasion'}]},
        {'text': "What's a fashionable outfit for a man in his 20s?", 'intent': 'get_attributes', 'entities': [{'start':  1, 'end': 3, 'value': 'male', 'entity': 'gender'}, {'start': 45, 'end': 48, 'value': '20s', 'entity': 'age'}, {'start': 9, 'end': 20, 'value': 'fashionable', 'entity': 'style'}]},
        {'text': 'Suggest an outfit for Diwali celebration for a female.', 'intent': 'get_attributes', 'entities': [{'start': 22, 'end': 28, 'value': 'Diwali', 'entity': 'occasion'}, {'start': 47, 'end': 53, 'value': 'female', 'entity': 'gender'}]},
        {'text': "What should I wear for Holi festival? I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 23, 'end': 27, 'value': 'Holi', 'entity': 'occasion'}, {'start': 44, 'end': 48, 'value': 'male', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for Eid celebration. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 27, 'value': 'Eid', 'entity': 'occasion'}, {'start': 47, 'end': 53, 'value': 'female', 'entity': 'gender'}]},
        {'text': "I need a traditional attire for Navratri festival. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 40, 'value': 'Navratri', 'entity': 'occasion'}, {'start': 57, 'end': 61, 'value': 'male', 'entity': 'gender'}]},
        {'text': "What's a suitable outfit for Ganesh Chaturthi? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 29, 'end': 45, 'value': 'Ganesh Chaturthi', 'entity': 'occasion'}, {'start': 53, 'end': 59, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Recommend a festive outfit for Pongal celebration. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 31, 'end': 37, 'value': 'Pongal', 'entity': 'occasion'}, {'start': 57, 'end': 61, 'value': 'male', 'entity': 'gender'}]},
        {'text': "I'm attending a Durga Puja event. What should I wear? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 26, 'value': 'Durga Puja', 'entity': 'occasion'}, {'start': 60, 'end': 66, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Suggest an outfit for Makar Sankranti festival. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 22, 'end': 37, 'value': 'Makar Sankranti', 'entity': 'occasion'}, {'start': 54, 'end': 58, 'value': 'male', 'entity': 'gender'}]},
        {'text': "What's a traditional attire for Onam celebration? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 36, 'value': 'Onam', 'entity': 'occasion'}, {'start': 56, 'end': 62, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for Baisakhi festival. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 32, 'value': 'Baisakhi', 'entity': 'occasion'}, {'start': 49, 'end': 53, 'value': 'male', 'entity': 'gender'}]},
        {'text': "I need a festive attire for Janmashtami celebration. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 28, 'end': 39, 'value': 'Janmashtami', 'entity': 'occasion'}, {'start': 59, 'end': 65, 'value': 'female', 'entity': 'gender'}]},
        {'text': "What should I wear for Karva Chauth? I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 23, 'end': 35, 'value': 'Karva Chauth', 'entity': 'occasion'}, {'start': 43, 'end': 47, 'value': 'male', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for Gudi Padwa festival. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 34, 'value': 'Gudi Padwa', 'entity': 'occasion'}, {'start': 51, 'end': 57, 'value': 'female', 'entity': 'gender'}]},
        {'text': "I'm attending a Dussehra event. What should I wear? I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 24, 'value': 'Dussehra', 'entity': 'occasion'}, {'start': 58, 'end': 62, 'value': 'male', 'entity': 'gender'}]},
        {'text': "Suggest an outfit for Eid ul Fitr celebration. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 22, 'end': 33, 'value': 'Eid ul Fitr', 'entity': 'occasion'}, {'start': 53, 'end': 59, 'value': 'female', 'entity': 'gender'}]},
        {'text': "What's a traditional attire for Makar Sankranti? I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 47, 'value': 'Makar Sankranti', 'entity': 'occasion'}, {'start': 55, 'end': 59, 'value': 'male', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for Lohri festival. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 29, 'value': 'Lohri', 'entity': 'occasion'}, {'start': 46, 'end': 52, 'value': 'female', 'entity': 'gender'}]},
        {'text': "I need a festive attire for Raksha Bandhan celebration. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 28, 'end': 42, 'value': 'Raksha Bandhan', 'entity': 'occasion'}, {'start': 62, 'end': 66, 'value': 'male', 'entity': 'gender'}]},
        {'text': "What's a suitable outfit for Christmas party? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 29, 'end': 38, 'value': 'Christmas', 'entity': 'occasion'}, {'start': 52, 'end': 58, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for New Year's Eve celebration. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 38, 'value': "New Year's Eve", 'entity': 'occasion'}, {'start': 58, 'end': 62, 'value': 'male', 'entity': 'gender'}]},
        {'text': "I'm attending a Halloween party. What should I wear? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 25, 'value': 'Halloween', 'entity': 'occasion'}, {'start': 59, 'end': 65, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Suggest an outfit for Valentine's Day. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 22, 'end': 37, 'value': "Valentine's Day", 'entity': 'occasion'}, {'start': 45, 'end': 49, 'value': 'male', 'entity': 'gender'}]},
        {'text': "What's a festive outfit for Thanksgiving dinner? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 28, 'end': 40, 'value': 'Thanksgiving', 'entity': 'occasion'}, {'start': 55, 'end': 61, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for St. Patrick's Day. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 41, 'value': "St. Patrick's Day", 'entity': 'occasion'}, {'start': 49, 'end': 53, 'value': 'male', 'entity': 'gender'}]},
        {'text': "I need a stylish attire for Chinese New Year. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 28, 'end': 44, 'value': 'Chinese New Year', 'entity': 'occasion'}, {'start': 52, 'end': 58, 'value': 'female', 'entity': 'gender'}]},
        {'text': "What should I wear for Independence Day celebration? I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 23, 'end': 39, 'value': 'Independence Day', 'entity': 'occasion'}, {'start': 59, 'end': 63, 'value': 'male', 'entity': 'gender'}]},
        {'text': "Suggest an outfit for Easter brunch. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 22, 'end': 28, 'value': 'Easter', 'entity': 'occasion'}, {'start': 43, 'end': 49, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for Oktoberfest. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 35, 'value': 'Oktoberfest', 'entity': 'occasion'}, {'start': 43, 'end': 47, 'value': 'male', 'entity': 'gender'}]},
        {'text': "I'm attending a Mardi Gras parade. What should I wear? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 26, 'value': 'Mardi Gras', 'entity': 'occasion'}, {'start': 61, 'end': 67, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Suggest an outfit for Bastille Day. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 22, 'end': 34, 'value': 'Bastille Day', 'entity': 'occasion'}, {'start': 42, 'end': 46, 'value': 'male', 'entity': 'gender'}]},
        {'text': "What's a festive attire for Eid al Adha? I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 28, 'end': 39, 'value': 'Eid al Adha', 'entity': 'occasion'}, {'start': 47, 'end': 53, 'value': 'female', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for Canada Day celebration. I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 34, 'value': 'Canada Day', 'entity': 'occasion'}, {'start': 54, 'end': 58, 'value': 'male', 'entity': 'gender'}]},
        {'text': "I need a stylish attire for Diwali festival. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 28, 'end': 34, 'value': 'Diwali', 'entity': 'occasion'}, {'start': 51, 'end': 57, 'value': 'female', 'entity': 'gender'}]},
        {'text': "What should I wear for Hanukkah celebration? I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 23, 'end': 31, 'value': 'Hanukkah', 'entity': 'occasion'}, {'start': 51, 'end': 55, 'value': 'male', 'entity': 'gender'}]},
        {'text': "Suggest an outfit for Bastille Day. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 22, 'end': 34, 'value': 'Bastille Day', 'entity': 'occasion'}, {'start': 42, 'end': 48, 'value': 'female', 'entity': 'gender'}]},
        {'text': "I'm attending a Nowruz celebration. What should I wear? I'm a male.", 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 22, 'value': 'Nowruz', 'entity': 'occasion'}, {'start': 62, 'end': 66, 'value': 'male', 'entity': 'gender'}]},
        {'text': "Recommend an outfit for Songkran festival. I'm a female.", 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 32, 'value': 'Songkran', 'entity': 'occasion'}, {'start': 49, 'end': 55, 'value': 'female', 'entity': 'gender'}]},
        {'text': "I'm looking for a casual outfit in blue and white colors.", 'intent': 'get_attributes', 'entities': [{'start': 35, 'end': 39, 'value': 'blue', 'entity': 'color'}, {'start': 44, 'end': 49, 'value': 'white', 'entity': 'color'}, {'start': 18, 'end': 24, 'value': 'casual', 'entity': 'style'}]},
        {'text': 'Recommend a formal attire in black for the event.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'formal', 'entity': 'style'}, {'start': 29, 'end': 34, 'value': 'black', 'entity': 'color'}]},
        {'text': 'Suggest a fancy dress with gold accents.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 15, 'value': 'fancy', 'entity': 'style'}, {'start': 27, 'end': 31, 'value': 'gold', 'entity': 'color'}]},
        {'text': 'I need a dress in pastel shades like pink and lavender.', 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 24, 'value': 'pastel', 'entity': 'style'}, {'start': 37, 'end': 41, 'value': 'pink', 'entity': 'color'}, {'start': 46, 'end': 54, 'value': 'lavender', 'entity': 'color'}]},
        {'text': 'Recommend a floral outfit in light colors.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'floral', 'entity': 'style'}, {'start': 29, 'end': 34, 'value': 'light', 'entity': 'color'}]},
        {'text': "I'm looking for a chic outfit with monochromatic colors.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 22, 'value': 'chic', 'entity': 'style'}, {'start': 35, 'end': 48, 'value': 'monochromatic', 'entity': 'color'}]},
        {'text': 'Suggest a modern ensemble with neutral shades.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 16, 'value': 'modern', 'entity': 'style'}, {'start': 31, 'end': 38, 'value': 'neutral', 'entity': 'color'}]},
        {'text': 'I need a vibrant dress in bold colors for the party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 16, 'value': 'vibrant', 'entity': 'style'}, {'start': 26, 'end': 30, 'value': 'bold', 'entity': 'color'}, {'start': 31, 'end': 37, 'value': 'colors', 'entity': 'color'}]},
        {'text': 'Recommend a minimalist outfit with earthy tones.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 22, 'value': 'minimalist', 'entity': 'style'}, {'start': 35, 'end': 47, 'value': 'earthy tones', 'entity': 'color'}]},
        {'text': 'Suggest a bohemian dress with vibrant prints.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 18, 'value': 'bohemian', 'entity': 'style'}, {'start': 30, 'end': 37, 'value': 'vibrant', 'entity': 'color'}, {'start': 38, 'end': 44, 'value': 'prints', 'entity': 'style'}]},
        {'text': 'I need a retro outfit with classic color combinations.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'retro', 'entity': 'style'}, {'start': 27, 'end': 34, 'value': 'classic', 'entity': 'style'}, {'start': 35, 'end': 53, 'value': 'color combinations', 'entity': 'color'}]},
        {'text': 'Recommend a sporty ensemble in bold shades.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'sporty', 'entity': 'style'}, {'start': 31, 'end': 35, 'value': 'bold', 'entity': 'color'}, {'start': 36, 'end': 42, 'value': 'shades', 'entity': 'color'}]},
        {'text': 'Suggest a glamorous dress with glittering details.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 19, 'value': 'glamorous', 'entity': 'style'}, {'start': 31, 'end': 41, 'value': 'glittering', 'entity': 'style'}, {'start': 42, 'end': 49, 'value': 'details', 'entity': 'style'}]},
        {'text': "I'm looking for a vintage outfit with muted colors.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 25, 'value': 'vintage', 'entity': 'style'}, {'start': 38, 'end': 43, 'value': 'muted', 'entity': 'color'}]},
        {'text': 'Recommend a preppy dress in bright and cheerful colors.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'preppy', 'entity': 'style'}, {'start': 28, 'end': 34, 'value': 'bright', 'entity': 'color'}, {'start': 39, 'end': 47, 'value': 'cheerful', 'entity': 'color'}]},
        {'text': 'Suggest an elegant ensemble with soft pastel hues.', 'intent': 'get_attributes', 'entities': [{'start': 11, 'end': 18, 'value': 'elegant', 'entity': 'style'}, {'start': 33, 'end': 37, 'value': 'soft', 'entity': 'color'}, {'start': 38, 'end': 44, 'value': 'pastel', 'entity': 'color'}]},
        {'text': 'I need a modern outfit with monochromatic colors.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'modern', 'entity': 'style'}, {'start': 28, 'end': 41, 'value': 'monochromatic', 'entity': 'color'}]},
        {'text': 'Recommend a casual dress with earthy tones.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'casual', 'entity': 'style'}, {'start': 30, 'end': 42, 'value': 'earthy tones', 'entity': 'color'}]},
        {'text': 'Suggest a vibrant ensemble with bold shades.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 17, 'value': 'vibrant', 'entity': 'style'}, {'start': 32, 'end': 36, 'value': 'bold', 'entity': 'color'}]},
        {'text': "I'm looking for a bohemian outfit with colorful prints.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 26, 'value': 'bohemian', 'entity': 'style'}, {'start': 39, 'end': 47, 'value': 'colorful', 'entity': 'color'}, {'start': 48, 'end': 54, 'value': 'prints', 'entity': 'style'}]},
        {'text': "I'm looking for a casual outfit in blue and white colors.", 'intent': 'get_attributes', 'entities': [{'start': 35, 'end': 39, 'value': 'blue', 'entity': 'color'}, {'start': 44, 'end': 49, 'value': 'white', 'entity': 'color'}, {'start': 18, 'end': 24, 'value': 'casual', 'entity': 'style'}]},
        {'text': 'Recommend a formal attire in black for the event.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'formal', 'entity': 'style'}, {'start': 29, 'end': 34, 'value': 'black', 'entity': 'color'}]},
        {'text': 'Suggest a fancy dress with gold accents.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 15, 'value': 'fancy', 'entity': 'style'}, {'start': 27, 'end': 31, 'value': 'gold', 'entity': 'color'}]},
        {'text': 'I need a dress in pastel shades like pink and lavender.', 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 24, 'value': 'pastel', 'entity': 'style'}, {'start': 37, 'end': 41, 'value': 'pink', 'entity': 'color'}, {'start': 46, 'end': 54, 'value': 'lavender', 'entity': 'color'}]},
        {'text': 'Recommend a floral outfit in light colors.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'floral', 'entity': 'style'}, {'start': 29, 'end': 34, 'value': 'light', 'entity': 'color'}]},
        {'text': "I'm looking for a chic outfit with monochromatic colors.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 22, 'value': 'chic', 'entity': 'style'}, {'start': 35, 'end': 48, 'value': 'monochromatic', 'entity': 'color'}]},
        {'text': 'Suggest a modern ensemble with neutral shades.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 16, 'value': 'modern', 'entity': 'style'}, {'start': 31, 'end': 38, 'value': 'neutral', 'entity': 'color'}]},
        {'text': 'I need a vibrant dress in bold colors for the party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 16, 'value': 'vibrant', 'entity': 'style'}, {'start': 26, 'end': 30, 'value': 'bold', 'entity': 'color'}, {'start': 31, 'end': 37, 'value': 'colors', 'entity': 'color'}]},
        {'text': 'Recommend a minimalist outfit with earthy tones.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 22, 'value': 'minimalist', 'entity': 'style'}, {'start': 35, 'end': 47, 'value': 'earthy tones', 'entity': 'color'}]},
        {'text': 'Suggest a bohemian dress with vibrant prints.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 18, 'value': 'bohemian', 'entity': 'style'}, {'start': 30, 'end': 37, 'value': 'vibrant', 'entity': 'color'}, {'start': 38, 'end': 44, 'value': 'prints', 'entity': 'style'}]},
        {'text': 'I need a retro outfit with classic color combinations.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'retro', 'entity': 'style'}, {'start': 27, 'end': 34, 'value': 'classic', 'entity': 'style'}, {'start': 35, 'end': 53, 'value': 'color combinations', 'entity': 'color'}]},
        {'text': 'Recommend a sporty ensemble in bold shades.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'sporty', 'entity': 'style'}, {'start': 31, 'end': 35, 'value': 'bold', 'entity': 'color'}, {'start': 36, 'end': 42, 'value': 'shades', 'entity': 'color'}]},
        {'text': 'Suggest a glamorous dress with glittering details.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 19, 'value': 'glamorous', 'entity': 'style'}, {'start': 31, 'end': 41, 'value': 'glittering', 'entity': 'style'}, {'start': 42, 'end': 49, 'value': 'details', 'entity': 'style'}]},
        {'text': "I'm looking for a vintage outfit with muted colors.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 25, 'value': 'vintage', 'entity': 'style'}, {'start': 38, 'end': 43, 'value': 'muted', 'entity': 'color'}]},
        {'text': 'Recommend a preppy dress in bright and cheerful colors.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'preppy', 'entity': 'style'}, {'start': 28, 'end': 34, 'value': 'bright', 'entity': 'color'}, {'start': 39, 'end': 47, 'value': 'cheerful', 'entity': 'color'}]},
        {'text': 'Suggest an elegant ensemble with soft pastel hues.', 'intent': 'get_attributes', 'entities': [{'start': 11, 'end': 18, 'value': 'elegant', 'entity': 'style'}, {'start': 33, 'end': 37, 'value': 'soft', 'entity': 'color'}, {'start': 38, 'end': 44, 'value': 'pastel', 'entity': 'color'}]},
        {'text': 'I need a modern outfit with monochromatic colors.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'modern', 'entity': 'style'}, {'start': 28, 'end': 41, 'value': 'monochromatic', 'entity': 'color'}]},
        {'text': 'Recommend a casual dress with earthy tones.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 18, 'value': 'casual', 'entity': 'style'}, {'start': 30, 'end': 42, 'value': 'earthy tones', 'entity': 'color'}]},
        {'text': 'Suggest a vibrant ensemble with bold shades.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 17, 'value': 'vibrant', 'entity': 'style'}, {'start': 32, 'end': 36, 'value': 'bold', 'entity': 'color'}]},
        {'text': "I'm looking for a bohemian outfit with colorful prints.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 26, 'value': 'bohemian', 'entity': 'style'}, {'start': 39, 'end': 47, 'value': 'colorful', 'entity': 'color'}, {'start': 48, 'end': 54, 'value': 'prints', 'entity': 'style'}]},
        {'text': 'Can you recommend the same dress with full sleeves instead?', 'intent': 'get_attributes', 'entities': [{'start': 38, 'end': 50, 'value': 'full sleeves', 'entity': 'sleeve'}]},
        {'text': 'I like the outfit, but can you change it to short sleeves?', 'intent': 'get_attributes', 'entities': [{'start': 44, 'end': 57, 'value': 'short sleeves', 'entity': 'sleeve'}]},
        {'text': 'Is there an option for this dress with no sleeves?', 'intent': 'get_attributes', 'entities': [{'start': 39, 'end': 49, 'value': 'no sleeves', 'entity': 'sleeve'}]},
        {'text': "I'd like this top with three quarter sleeves instead.", 'intent': 'get_attributes', 'entities': [{'start': 23, 'end': 44, 'value': 'three quarter sleeves', 'entity': 'sleeve'}]},
        {'text': 'Can you change the sleeves of this dress to long ones?', 'intent': 'get_attributes', 'entities': [{'start': 44, 'end': 53, 'value': 'long ones', 'entity': 'sleeve'}]},
        {'text': "I'd prefer a sleeveless version of this outfit.", 'intent': 'get_attributes', 'entities': [{'start': 13, 'end': 23, 'value': 'sleeveless', 'entity': 'sleeve'}]},
        {'text': 'Is there a similar dress with cap sleeves available?', 'intent': 'get_attributes', 'entities': [{'start': 30, 'end': 41, 'value': 'cap sleeves', 'entity': 'sleeve'}]},
        {'text': 'Could you change the dress to a halter neck style?', 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 43, 'value': 'halter neck', 'entity': 'sleeve'}]},
        {'text': 'Can you show me the same top with butterfly sleeves?', 'intent': 'get_attributes', 'entities': [{'start': 34, 'end': 51, 'value': 'butterfly sleeves', 'entity': 'sleeve'}]},
        {'text': "I'd like to see this blouse with puffed sleeves instead.", 'intent': 'get_attributes', 'entities': [{'start': 33, 'end': 47, 'value': 'puffed sleeves', 'entity': 'sleeve'}]},
        {'text': 'Can you show me this dress with matching bangles?', 'intent': 'get_attributes', 'entities': [{'start': 41, 'end': 48, 'value': 'bangles', 'entity': 'handwear'}]},
        {'text': 'Do you have any bracelets that would go well with this outfit?', 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 25, 'value': 'bracelets', 'entity': 'handwear'}]},
        {'text': "I'd like to see how this top looks with a matching ring.", 'intent': 'get_attributes', 'entities': [{'start': 51, 'end': 55, 'value': 'ring', 'entity': 'handwear'}]},
        {'text': 'Can you recommend some handwear to go with this dress?', 'intent': 'get_attributes', 'entities': [{'start': 23, 'end': 31, 'value': 'handwear', 'entity': 'handwear'}]},
        {'text': "I'm looking for a dress that can be paired with elegant bangles.", 'intent': 'get_attributes', 'entities': [{'start': 56, 'end': 63, 'value': 'bangles', 'entity': 'handwear'}]},
        {'text': 'Could you show me options for bracelets to go with this outfit?', 'intent': 'get_attributes', 'entities': [{'start': 30, 'end': 39, 'value': 'bracelets', 'entity': 'handwear'}]},
        {'text': "I'd like to see this gown with a statement ring, if possible.", 'intent': 'get_attributes', 'entities': [{'start': 43, 'end': 47, 'value': 'ring', 'entity': 'handwear'}]},
        {'text': 'Do you have any matching handwear for this saree?', 'intent': 'get_attributes', 'entities': [{'start': 25, 'end': 33, 'value': 'handwear', 'entity': 'handwear'}]},
        {'text': 'I want to try this kurta with some stylish bangles.', 'intent': 'get_attributes', 'entities': [{'start': 43, 'end': 50, 'value': 'bangles', 'entity': 'handwear'}]},
        {'text': 'Are there any rings available that would match this dress?', 'intent': 'get_attributes', 'entities': [{'start': 14, 'end': 19, 'value': 'rings', 'entity': 'handwear'}]},
        {'text': 'Show me some dresses made of georgette for a summer party.', 'intent': 'get_attributes', 'entities': [{'start': 29, 'end': 38, 'value': 'georgette', 'entity': 'material'}, {'start': 52, 'end': 57, 'value': 'party', 'entity': 'occasion'}, {'start': 45, 'end': 51, 'value': 'summer', 'entity': 'weather'}]},
        {'text': "I'm looking for a chiffon outfit in light colors for a casual occasion.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 25, 'value': 'chiffon', 'entity': 'material'}, {'start': 55, 'end': 61, 'value': 'casual', 'entity': 'occasion'}]},
        {'text': 'Can you suggest a dress made of silk for a formal event?', 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 36, 'value': 'silk', 'entity': 'material'}, {'start': 43, 'end': 49, 'value': 'formal', 'entity': 'occasion'}]},
        {'text': "I'd like to see outfits made of nylon for rainy weather.", 'intent': 'get_attributes', 'entities': [{'start': 32, 'end': 37, 'value': 'nylon', 'entity': 'material'}, {'start': 42, 'end': 47, 'value': 'rainy', 'entity': 'weather'}]},
        {'text': 'Do you have any synthetic material dresses for a party?', 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 25, 'value': 'synthetic', 'entity': 'material'}, {'start': 49, 'end': 54, 'value': 'party', 'entity': 'occasion'}]},
        {'text': 'Show me cotton outfits for a casual outing.', 'intent': 'get_attributes', 'entities': [{'start': 8, 'end': 14, 'value': 'cotton', 'entity': 'material'}, {'start': 29, 'end': 35, 'value': 'casual', 'entity': 'occasion'}]},
        {'text': "I'm interested in dresses made of georgette or silk for an elegant event.", 'intent': 'get_attributes', 'entities': [{'start': 34, 'end': 43, 'value': 'georgette', 'entity': 'material'}, {'start': 47, 'end': 51, 'value': 'silk', 'entity': 'material'}, {'start': 67, 'end': 72, 'value': 'event', 'entity': 'occasion'}]},
        {'text': 'Can you suggest a synthetic material dress for a night out?', 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 27, 'value': 'synthetic', 'entity': 'material'}, {'start': 49, 'end': 58, 'value': 'night out', 'entity': 'occasion'}]},
        {'text': 'Show me outfits made of nylon that are suitable for winter.', 'intent': 'get_attributes', 'entities': [{'start': 24, 'end': 29, 'value': 'nylon', 'entity': 'material'}, {'start': 52, 'end': 58, 'value': 'winter', 'entity': 'weather'}]},
        {'text': "I'd like to see chiffon dresses for a beach vacation.", 'intent': 'get_attributes', 'entities': [{'start': 16, 'end': 23, 'value': 'chiffon', 'entity': 'material'}, {'start': 44, 'end': 52, 'value': 'vacation', 'entity': 'occasion'}]},
        {'text': 'I need a light blue cotton shirt for a hot summer day.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 19, 'value': 'light blue', 'entity': 'color'}, {'start': 20, 'end': 26, 'value': 'cotton', 'entity': 'material'}, {'start': 27, 'end': 32, 'value': 'shirt', 'entity': 'top_type'}, {'start': 43, 'end': 48, 'value': 'summer', 'entity': 'weather'}]},
        {'text': 'Show me a fancy dress with floral pattern for a garden party.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 15, 'value': 'fancy', 'entity': 'style'}, {'start': 27, 'end': 41, 'value': 'floral pattern', 'entity': 'style'}, {'start': 55, 'end': 60, 'value': 'party', 'entity': 'occasion'}]},
        {'text': "I'm looking for a leather jacket in black color for a cool evening.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 25, 'value': 'leather', 'entity': 'material'}, {'start': 36, 'end': 41, 'value': 'black', 'entity': 'color'}, {'start': 26, 'end': 32, 'value': 'jacket', 'entity': 'top_type'}, {'start': 54, 'end': 66, 'value': 'cool evening', 'entity': 'weather'}]},
        {'text': 'Can you suggest a chiffon frock with plain design for a formal event?', 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 25, 'value': 'chiffon', 'entity': 'material'}, {'start': 26, 'end': 31, 'value': 'frock', 'entity': 'top_type'}, {'start': 37, 'end': 49, 'value': 'plain design', 'entity': 'style'}, {'start': 56, 'end': 62, 'value': 'formal', 'entity': 'occasion'}]},
        {'text': 'Show me a georgette frock with stripes for a semi formal gathering.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 19, 'value': 'georgette', 'entity': 'material'}, {'start': 20, 'end': 25, 'value': 'frock', 'entity': 'top_type'}, {'start': 31, 'end': 38, 'value': 'stripes', 'entity': 'style'}, {'start': 50, 'end': 56, 'value': 'formal', 'entity': 'occasion'}]},
        {'text': "I'd like to see a casual cotton shirt in plain white color.", 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 24, 'value': 'casual', 'entity': 'style'}, {'start': 25, 'end': 31, 'value': 'cotton', 'entity': 'material'}, {'start': 32, 'end': 37, 'value': 'shirt', 'entity': 'top_type'}, {'start': 41, 'end': 46, 'value': 'plain', 'entity': 'style'}, {'start': 47, 'end': 52, 'value': 'white', 'entity': 'color'}]},
        {'text': 'Show me a leather jacket with ankle strap heels for a trendy look.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 17, 'value': 'leather', 'entity': 'material'}, {'start': 18, 'end': 24, 'value': 'jacket', 'entity': 'top_type'}, {'start': 30, 'end': 41, 'value': 'ankle strap', 'entity': 'shoe_type'}, {'start': 42, 'end': 47, 'value': 'heels', 'entity': 'shoe_type'}, {'start': 54, 'end': 65, 'value': 'trendy look', 'entity': 'style'}]},
        {'text': 'I need a kurta made of silk with floral print for a traditional event.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'kurta', 'entity': 'top_type'}, {'start': 23, 'end': 27, 'value': 'silk', 'entity': 'material'}, {'start': 33, 'end': 45, 'value': 'floral print', 'entity': 'style'}, {'start': 52, 'end': 63, 'value': 'traditional', 'entity': 'occasion'}]},
        {'text': 'Can you suggest a satin top in bright colors for a night out?', 'intent': 'get_attributes', 'entities': [{'start': 18, 'end': 23, 'value': 'satin', 'entity': 'material'}, {'start': 24, 'end': 27, 'value': 'top', 'entity': 'top_type'}, {'start': 31, 'end': 37, 'value': 'bright', 'entity': 'color'}, {'start': 38, 'end': 44, 'value': 'colors', 'entity': 'color'}, {'start': 51, 'end': 60, 'value': 'night out', 'entity': 'occasion'}]},
        {'text': 'Show me a cotton shirt with stripes in earthy tones for a casual day.', 'intent': 'get_attributes', 'entities': [{'start': 10, 'end': 16, 'value': 'cotton', 'entity': 'material'}, {'start': 17, 'end': 22, 'value': 'shirt', 'entity': 'top_type'}, {'start': 28, 'end': 35, 'value': 'stripes', 'entity': 'style'}, {'start': 39, 'end': 51, 'value': 'earthy tones', 'entity': 'color'}, {'start': 58, 'end': 64, 'value': 'casual', 'entity': 'occasion'}]},
        {"text": "I need a blue leather purse with a flap and matching shoes.", "intent": "get_attributes", "entities": [{"start": 9, "end": 21, "value": "blue", "entity": "color"},{"start": 23, "end": 29, "value": "leather", "entity": "material"},{"start": 31, "end": 35, "value": "purse", "entity": "bag_type"},{"start": 40, "end": 44, "value": "flap", "entity": "top_type"},{"start": 51, "end": 57, "value": "matching", "entity": "color"},{"start": 58, "end": 63, "value": "shoes", "entity": "shoe_type"}]},
        {"text": "Looking for a red velvet bag with a zipper and black shoes.", "intent": "get_attributes", "entities": [{"start": 14, "end": 17, "value": "red", "entity": "color"},{"start": 19, "end": 25, "value": "velvet", "entity": "material"},{"start": 27, "end": 30, "value": "bag", "entity": "bag_type"},{"start": 35, "end": 41, "value": "zipper", "entity": "top_type"},{"start": 48, "end": 53, "value": "black", "entity": "color"},{"start": 54, "end": 59, "value": "shoes", "entity": "shoe_type"}]},
        {"text": "I want a brown leather backpack with pockets and white sneakers.", "intent": "get_attributes", "entities": [{"start": 9, "end": 15, "value": "brown", "entity": "color"},{"start": 17, "end": 23, "value": "leather", "entity": "material"},{"start": 25, "end": 33, "value": "backpack", "entity": "bag_type"},{"start": 38, "end": 45, "value": "pockets", "entity": "top_type"},{"start": 50, "end": 55, "value": "white", "entity": "color"},{"start": 56, "end": 64, "value": "sneakers", "entity": "shoe_type"}]},
        {"text": "Looking for a black leather hand bag with a handle and red shoes.", "intent": "get_attributes", "entities": [{"start": 14, "end": 18, "value": "black", "entity": "color"},{"start": 20, "end": 26, "value": "leather", "entity": "material"},{"start": 28, "end": 36, "value": "hand bag", "entity": "bag_type"},{"start": 41, "end": 47, "value": "handle", "entity": "top_type"},{"start": 54, "end": 57, "value": "red", "entity": "color"},{"start": 58, "end": 63, "value": "shoes", "entity": "shoe_type"}]},
        {"text": "I need a blue school bag with compartments and white sneakers.", "intent": "get_attributes", "entities": [{"start": 9, "end": 13, "value": "blue", "entity": "color"},{"start": 15, "end": 25, "value": "school bag", "entity": "bag_type"},{"start": 30, "end": 40, "value": "compartments", "entity": "feature"},{"start": 45, "end": 50, "value": "white", "entity": "color"},{"start": 51, "end": 58, "value": "sneakers", "entity": "shoe_type"}]},
        {
            "text": "She wore a stylish bodycon dress with a matching handbag.",
            "intent": "get_attributes",
            "entities": [
            {"start": 15, "end": 22, "value": "bodycon", "entity": "top_type"},
            {"start": 24, "end": 29, "value": "dress", "entity": "outfit_type"},
            {"start": 39, "end": 46, "value": "matching", "entity": "color"},
            {"start": 47, "end": 54, "value": "handbag", "entity": "bag_type"}
            ]
        },
        {
            "text": "She loves her weatherhiweather peplum top paired with jeans.",
            "intent": "get_attributes",
            "entities": [
            {"start": 12, "end": 18, "value": "peplum", "entity": "top_type"},
            {"start": 33, "end": 38, "value": "jeans", "entity": "bottom_type"}
            ]
        },
        {
            "text": "She wore a high waist skirt with a weatherute blouseon.",
            "intent": "get_attributes",
            "entities": [
            {"start": 10, "end": 20, "value": "high waist", "entity": "style"},
            {"start": 21, "end": 26, "value": "skirt", "entity": "bottom_type"},
            {"start": 32, "end": 36, "value": "cute", "entity": "style"},
            {"start": 37, "end": 45, "value": "blouseon", "entity": "outfit_type"}
            ]
        },
        {
            "text": "She rocked a cape over her t shirt for a stylish look.",
            "intent": "get_attributes",
            "entities": [
            {"start": 14, "end": 18, "value": "cape", "entity": "item_type"},
            {"start": 24, "end": 32, "value": "t shirt", "entity": "top_type"},
            {"start": 46, "end": 53, "value": "stylish", "entity": "style"}
            ]
        },
        {
            "text": "She looked stunning in her gypsy top and flowy skirt.",
            "intent": "get_attributes",
            "entities": [
            {"start": 23, "end": 28, "value": "gypsy", "entity": "top_type"},
            {"start": 38, "end": 43, "value": "flowy", "entity": "style"},
            {"start": 44, "end": 49, "value": "skirt", "entity": "bottom_type"}
            ]
        },
        {
            "text": "She paired her blazer with tailored pants for a professional look.",
            "intent": "get_attributes",
            "entities": [
            {"start": 10, "end": 16, "value": "blazer", "entity": "outfit_type"},
            {"start": 26, "end": 35, "value": "tailored", "entity": "style"},
            {"start": 41, "end": 46, "value": "pants", "entity": "bottom_type"},
            {"start": 52, "end": 63, "value": "professional", "entity": "style"}
            ]
        },
        {
            "text": "She chose a stunning ball gown for the gala event.",
            "intent": "get_attributes",
            "entities": [
            {"start": 17, "end": 27, "value": "ball gown", "entity": "outfit_type"},
            {"start": 31, "end": 35, "value": "gala", "entity": "occasion"},
            {"start": 36, "end": 41, "value": "event", "entity": "occasion"}
            ]
        },
        {
            "text": "She wore a stylish tunic with leggings for a comfortable look.",
            "intent": "get_attributes",
            "entities": [
            {"start": 10, "end": 16, "value": "tunic", "entity": "top_type"},
            {"start": 22, "end": 30, "value": "leggings", "entity": "bottom_type"},
            {"start": 44, "end": 54, "value": "comfortable", "entity": "style"}
            ]
        },
        {
            "text": "She rocked a one shoulder dress with confidence.",
            "intent": "get_attributes",
            "entities": [
            {"start": 10, "end": 22, "value": "one shoulder", "entity": "top_type"},
            {"start": 29, "end": 34, "value": "dress", "entity": "outfit_type"},
            {"start": 40, "end": 49, "value": "confidence", "entity": "style"}
            ]
        },
        {
            "text": "She loves her sheath dress for formal occasions.",
            "intent": "get_attributes",
            "entities": [
            {"start": 16, "end": 22, "value": "sheath", "entity": "top_type"},
            {"start": 23, "end": 29, "value": "dress", "entity": "outfit_type"},
            {"start": 33, "end": 39, "value": "formal", "entity": "occasion"}
            ]
        },
        {
            "text": "She prefers wearing a halter top during summer.",
            "intent": "get_attributes",
            "entities": [
            {"start": 24, "end": 30, "value": "halter", "entity": "top_type"},
            {"start": 42, "end": 48, "value": "summer", "entity": "weather"}
            ]
        },
        {
            "text": "She looked stunning in her chic gown at the party.",
            "intent": "get_attributes",
            "entities": [
            {"start": 25, "end": 29, "value": "chic", "entity": "style"},
            {"start": 33, "end": 37, "value": "gown", "entity": "outfit_type"},
            {"start": 41, "end": 46, "value": "party", "entity": "occasion"}
            ]
        },
        {
            "text": "She loves her fashionable frock for casual outings.",
            "intent": "get_attributes",
            "entities": [
            {"start": 16, "end": 21, "value": "fashionable", "entity": "style"},
            {"start": 22, "end": 27, "value": "frock", "entity": "outfit_type"},
            {"start": 31, "end": 38, "value": "casual", "entity": "occasion"}
            ]
        },
        {
            "text": "She rocked a bustier top with a skirt at the event.",
            "intent": "get_attributes",
            "entities": [
            {"start": 14, "end": 21, "value": "bustier", "entity": "top_type"},
            {"start": 36, "end": 41, "value": "skirt", "entity": "bottom_type"},
            {"start": 45, "end": 50, "value": "event", "entity": "occasion"}
            ]
        },
        {
            "text": "She looked elegant in her slip dress and heels.",
            "intent": "get_attributes",
            "entities": [
            {"start": 24, "end": 33, "value": "slip dress", "entity": "outfit_type"},
            {"start": 38, "end": 44, "value": "heels", "entity": "footwear"}
            ]
        },
        {
            "text": "She wore a pinafore dress with a blouse for the occasion.",
            "intent": "get_attributes",
            "entities": [
            {"start": 10, "end": 18, "value": "pinafore", "entity": "top_type"},
            {"start": 24, "end": 30, "value": "dress", "entity": "outfit_type"},
            {"start": 36, "end": 42, "value": "blouse", "entity": "top_type"},
            {"start": 46, "end": 55, "value": "occasion", "entity": "occasion"}
            ]
        },
        {
            "text": "She looked stunning in her empire dress at the event.",
            "intent": "get_attributes",
            "entities": [
            {"start": 27, "end": 33, "value": "empire", "entity": "style"},
            {"start": 34, "end": 40, "value": "dress", "entity": "outfit_type"},
            {"start": 44, "end": 49, "value": "event", "entity": "occasion"}
            ]
        },
        {
            "text": "She wore a tiered dress with confidence.",
            "intent": "get_attributes",
            "entities": [
            {"start": 10, "end": 17, "value": "tiered", "entity": "style"},
            {"start": 18, "end": 24, "value": "dress", "entity": "outfit_type"},
            {"start": 30, "end": 39, "value": "confidence", "entity": "style"}
            ]
        },
    {'text': 'I need a casual cotton dress for a summer party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'}, {'start': 42, 'end': 47, 'value': 'party', 'entity': 'occasion'}, {'start': 16, 'end': 22, 'value': 'cotton', 'entity': 'material'}, {'start': 23, 'end': 28, 'value': 'dress', 'entity': 'outfit_type'}, {'start': 35, 'end': 41, 'value': 'summer', 'entity': 'weather'}]},
    {'text': 'Looking for a fancy frock for a wedding.', 'intent': 'get_attributes', 'entities': [{'start': 14, 'end': 19, 'value': 'fancy', 'entity': 'style'}, {'start': 32, 'end': 39, 'value': 'wedding', 'entity': 'occasion'}, {'start': 20, 'end': 25, 'value': 'frock', 'entity': 'outfit_type'}]},
    {'text': 'I want a plain chiffon saree for a formal event.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 14, 'value': 'plain', 'entity': 'style'}, {'start': 35, 'end': 41, 'value': 'formal', 'entity': 'occasion'}, {'start': 15, 'end': 22, 'value': 'chiffon', 'entity': 'material'}, {'start': 23, 'end': 28, 'value': 'saree', 'entity': 'outfit_type'}]},
    {'text': "I'm searching for a casual shirt for beach vacation.", 'intent': 'get_attributes', 'entities': [{'start': 20, 'end': 26, 'value': 'casual', 'entity': 'style'}, {'start': 45, 'end': 53, 'value': 'vacation', 'entity': 'occasion'}, {'start': 27, 'end': 32, 'value': 'shirt', 'entity': 'outfit_type'}]},
    {'text': 'Looking for leather boots for a rainy day.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 19, 'value': 'leather', 'entity': 'material'}, {'start': 32, 'end': 37, 'value': 'rainy', 'entity': 'weather'}, {'start': 20, 'end': 25, 'value': 'boots', 'entity': 'shoe_type'}]},
    {'text': 'I need a georgette jacket for a party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 18, 'value': 'georgette', 'entity': 'material'}, {'start': 32, 'end': 37, 'value': 'party', 'entity': 'style'}, {'start': 19, 'end': 25, 'value': 'jacket', 'entity': 'outfit_type'}]},
    {'text': 'Looking for skinny jeans for a casual day out.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 17, 'value': 'skinny', 'entity': 'bottom_type'}, {'start': 30, 'end': 36, 'value': 'casual', 'entity': 'occasion'}, {'start': 18, 'end': 23, 'value': 'jeans', 'entity': 'bottom_type'}]},
    {'text': 'I want stirrup pants for my dance practice.', 'intent': 'get_attributes', 'entities': [{'start': 7, 'end': 15, 'value': 'stirrup', 'entity': 'bottom_type'}, {'start': 29, 'end': 43, 'value': 'dance practice', 'entity': 'occasion'}, {'start': 16, 'end': 21, 'value': 'pants', 'entity': 'bottom_type'}]},
    {'text': 'Looking for jeggings for a comfortable outfit.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 20, 'value': 'jeggings', 'entity': 'bottom_type'}]},
    {'text': 'I need leather pants for a stylish look.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 22, 'value': 'leather pant', 'entity': 'bottom_type'}, {'start': 32, 'end': 44, 'value': 'stylish look', 'entity': 'style'}]},
    {'text': 'Looking for a skirt for a summer day.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 17, 'value': 'skirt', 'entity': 'bottom_type'}, {'start': 29, 'end': 36, 'value': 'summer', 'entity': 'weather'}]},
    {'text': 'I want trousers for a formal event.', 'intent': 'get_attributes', 'entities': [{'start': 7, 'end': 15, 'value': 'trousers', 'entity': 'bottom_type'}, {'start': 25, 'end': 31, 'value': 'formal', 'entity': 'occasion'}]},
    {'text': 'Looking for pants for a casual outing.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 17, 'value': 'pants', 'entity': 'bottom_type'}, {'start': 29, 'end': 35, 'value': 'casual', 'entity': 'occasion'}]},
    {'text': 'I need bell bottoms for a retro themed party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 21, 'value': 'bell bottoms', 'entity': 'bottom_type'}, {'start': 45, 'end': 50, 'value': 'party', 'entity': 'occasion'}]},
    {'text': 'Looking for straight pants for a formal look.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 20, 'value': 'straight', 'entity': 'bottom_type'}, {'start': 30, 'end': 41, 'value': 'formal look', 'entity': 'style'}, {'start': 21, 'end': 26, 'value': 'pants', 'entity': 'bottom_type'}]},
    {'text': 'I want a gown for the red carpet event.', 'intent': 'get_attributes', 'entities': [{'start': 7, 'end': 11, 'value': 'gown', 'entity': 'bottom_type'}, {'start': 37, 'end': 42, 'value': 'event', 'entity': 'occasion'}]},
    {'text': 'Looking for loose pants for a comfortable day.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 23, 'value': 'loose pant', 'entity': 'bottom_type'}, {'start': 35, 'end': 50, 'value': 'comfortable day', 'entity': 'occasion'}]},
    {'text': 'Looking for a stylish cabas bag for everyday use.', 'intent': 'get_attributes', 'entities': [{'start': 26, 'end': 30, 'value': 'cabas', 'entity': 'bag_type'}, {'start': 42, 'end': 51, 'value': 'everyday', 'entity': 'occasion'}, {'start': 31, 'end': 34, 'value': 'bag', 'entity': 'outfit_type'}]},
    {'text': 'I want a shoulder bag for a casual outing.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 18, 'value': 'shoulder', 'entity': 'bag_type'}, {'start': 30, 'end': 36, 'value': 'casual', 'entity': 'occasion'}, {'start': 19, 'end': 22, 'value': 'bag', 'entity': 'outfit_type'}]},
    {'text': 'Looking for a back pack for my hiking trip.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 20, 'value': 'back pack', 'entity': 'bag_type'}, {'start': 37, 'end': 41, 'value': 'trip', 'entity': 'occasion'}]},
    {'text': 'Looking for a sling bag for a casual day out.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 16, 'value': 'sling', 'entity': 'bag_type'}, {'start': 28, 'end': 34, 'value': 'casual', 'entity': 'occasion'}, {'start': 17, 'end': 20, 'value': 'bag', 'entity': 'outfit_type'}]},
    {'text': 'I want a baguette bag for a retro themed party.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 17, 'value': 'baguette', 'entity': 'bag_type'}, {'start': 55, 'end': 58, 'value': 'party', 'entity': 'occasion'}]},
    {'text': 'Looking for a hobo bag for a comfortable day.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 15, 'value': 'hobo', 'entity': 'bag_type'}, {'start': 27, 'end': 42, 'value': 'comfortable day', 'entity': 'occasion'}, {'start': 16, 'end': 19, 'value': 'bag', 'entity': 'outfit_type'}]},
    {'text': 'I need a mini bag for a night out.', 'intent': 'get_attributes', 'entities': [{'start': 9, 'end': 13, 'value': 'mini', 'entity': 'bag_type'}, {'start': 27, 'end': 36, 'value': 'night out', 'entity': 'occasion'}, {'start': 14, 'end': 17, 'value': 'bag', 'entity': 'outfit_type'}]},
    {'text': 'Looking for a jute bag for an eco friendly event.', 'intent': 'get_attributes', 'entities': [{'start': 12, 'end': 16, 'value': 'jute', 'entity': 'bag_type'}, {'start': 40, 'end': 45, 'value': 'event', 'entity': 'occasion'}, {'start': 17, 'end': 20, 'value': 'bag', 'entity': 'outfit_type'}]},
    {'text': 'Looking for ankle strap heels for a formal event.', 'intent': 'get_attributes', 'entities': [{'start': 21, 'end': 32, 'value': 'ankle strap', 'entity': 'shoe_type'}, {'start': 33, 'end': 38, 'value': 'heels', 'entity': 'shoe_type'}, {'start': 43, 'end': 49, 'value': 'formal', 'entity': 'occasion'}]},
    {'text': "I want comfortable flat shoes for everyday wear.", 'intent': "get_attributes", "entities": [{"start": 21, "end": 25, "value": "flat", "entity": "shoe_type"}, {"start": 26, "end": 31, "value": "shoes", "entity": "outfit_type"}, {"start": 43, "end": 51, "value": "everyday", "entity": "occasion"}, {"start": 9, "end": 19, "value": "comfortable", "entity": "style"}]},
    {'text': "Looking for stylish sandals with a stiletto heel.", 'intent': "get_attributes", "entities": [{"start": 21, "end": 28, "value": "stylish", "entity": "style"}, {"start": 29, "end": 36, "value": "sandals", "entity": "shoe_type"}, {"start": 42, "end": 50, "value": "stiletto", "entity": "shoe_style"}, {"start": 51, "end": 56, "value": "heel", "entity": "shoe_part"}]},
    {'text': "Looking for comfortable loafers for casual wear.", 'intent': "get_attributes", "entities": [{"start": 21, "end": 28, "value": "comfortable", "entity": "style"}, {"start": 29, "end": 36, "value": "loafers", "entity": "shoe_type"}, {"start": 40, "end": 46, "value": "casual", "entity": "occasion"}]},
    {'text': "I need knee high boots for the winter season.", 'intent': "get_attributes", "entities": [{"start": 15, "end": 20, "value": "knee high", "entity": "shoe_type"}, {"start": 21, "end": 26, "value": "boots", "entity": "shoe_style"}, {"start": 31, "end": 37, "value": "winter", "entity": "weather"}]},
    {'text': "I'm looking for platform shoes for a retro themed party.", 'intent': "get_attributes", "entities": [{"start": 23, "end": 31, "value": "platform", "entity": "shoe_type"}, {"start": 32, "end": 37, "value": "shoes", "entity": "outfit_type"}, {"start": 53, "end": 58, "value": "party", "entity": "occasion"}]},
    {'text': "Searching for Mary Jane heels for a special occasion.", 'intent': "get_attributes", "entities": [{"start": 18, "end": 27, "value": "Mary Jane", "entity": "shoe_type"}, {"start": 28, "end": 33, "value": "heels", "entity": "shoe_style"}, {"start": 37, "end": 44, "value": "special", "entity": "occasion"}]},
    {'text': "I want open toe sandals for a beach vacation.", 'intent': "get_attributes", "entities": [{"start": 12, "end": 20, "value": "open toe", "entity": "shoe_type"}, {"start": 21, "end": 29, "value": "sandals", "entity": "shoe_style"}, {"start": 49, "end": 57, "value": "vacation", "entity": "occasion"}]},
    {'text': "I'm looking for peep toe heels for a night out.", 'intent': "get_attributes", "entities": [{"start": 23, "end": 31, "value": "peep toe", "entity": "shoe_type"}, {"start": 32, "end": 37, "value": "heels", "entity": "shoe_style"}, {"start": 41, "end": 50, "value": "night out", "entity": "occasion"}]},
    {'text': "I want sports shoes for my outdoor activities.", 'intent': "get_attributes", "entities": [{"start": 9, "end": 15, "value": "sports", "entity": "shoe_type"}, {"start": 16, "end": 21, "value": "shoes", "entity": "outfit_type"}, {"start": 26, "end": 42, "value": "outdoor activities", "entity": "occasion"}]},
    {'text': "Looking for sharp cone heels for a formal event.", 'intent': "get_attributes", "entities": [{"start": 21, "end": 25, "value": "sharp", "entity": "style"}, {"start": 26, "end": 30, "value": "cone", "entity": "shoe_type"}, {"start": 31, "end": 36, "value": "heels", "entity": "shoe_style"}, {"start": 41, "end": 47, "value": "formal", "entity": "occasion"}]},
    {'text': "I need slingback shoes for a summer party.", 'intent': "get_attributes", "entities": [{"start": 12, "end": 20, "value": "slingback", "entity": "shoe_type"}, {"start": 21, "end": 26, "value": "shoes", "entity": "outfit_type"}, {"start": 40, "end": 45, "value": "party", "entity": "occasion"}, {'start': 33, 'end': 41, 'value': 'summer', 'entity': 'weather'}]},
    {'text': "I want pumps for a formal look.", 'intent': "get_attributes", "entities": [{"start": 9, "end": 14, "value": "pumps", "entity": "shoe_type"}, {"start": 19, "end": 30, "value": "formal look", "entity": "style"}]},
    {'text': 'I need a stylish watch for a formal event.', 'intent': 'get_attributes', 'entities': [
        {'start': 15, 'end': 20, 'value': 'stylish', 'entity': 'style'},
        {'start': 21, 'end': 26, 'value': 'watch', 'entity': 'handwear'},
        {'start': 30, 'end': 36, 'value': 'formal', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a sports watch for my outdoor activities.', 'intent': 'get_attributes', 'entities': [
        {'start': 12, 'end': 18, 'value': 'sports', 'entity': 'style'},
        {'start': 19, 'end': 24, 'value': 'watch', 'entity': 'handwear'},
        {'start': 29, 'end': 45, 'value': 'outdoor activities', 'entity': 'occasion'},
    ]},
    {'text': 'I want a light cotton dress for a casual day.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 20, 'value': 'light cotton', 'entity': 'material'},
        {'start': 27, 'end': 32, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 37, 'end': 43, 'value': 'casual', 'entity': 'occasion'},
    ]},
    {'text': 'I need a fur coat for the winter season.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 12, 'value': 'fur', 'entity': 'material'},
        {'start': 13, 'end': 17, 'value': 'coat', 'entity': 'outfit_type'},
        {'start': 32, 'end': 38, 'value': 'winter', 'entity': 'weather'},
    ]},
    {'text': 'Looking for a polyster blouse for a formal event.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 28, 'value': 'polyster', 'entity': 'material'},
        {'start': 29, 'end': 35, 'value': 'blouse', 'entity': 'outfit_type'},
        {'start': 40, 'end': 46, 'value': 'formal', 'entity': 'occasion'},
    ]},
    {'text': "I'm searching for suede shoes for a classy look.", 'intent': 'get_attributes', 'entities': [
        {'start': 23, 'end': 28, 'value': 'suede', 'entity': 'material'},
        {'start': 29, 'end': 34, 'value': 'shoes', 'entity': 'outfit_type'},
        {'start': 39, 'end': 50, 'value': 'classy look', 'entity': 'style'},
    ]},
    {'text': 'I want a jersey dress for a casual outing.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 15, 'value': 'jersey', 'entity': 'material'},
        {'start': 22, 'end': 27, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 32, 'end': 38, 'value': 'casual', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a twill jacket for a trendy look.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 26, 'value': 'twill', 'entity': 'material'},
        {'start': 27, 'end': 34, 'value': 'jacket', 'entity': 'outfit_type'},
        {'start': 39, 'end': 50, 'value': 'trendy look', 'entity': 'style'},
    ]},
    {'text': 'I need a banian shirt for a relaxed day.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 15, 'value': 'banian', 'entity': 'material'},
        {'start': 16, 'end': 21, 'value': 'shirt', 'entity': 'outfit_type'},
        {'start': 26, 'end': 37, 'value': 'relaxed day', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a wool sweater for chilly weather.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 25, 'value': 'wool', 'entity': 'material'},
        {'start': 26, 'end': 33, 'value': 'sweater', 'entity': 'outfit_type'},
        {'start': 38, 'end': 51, 'value': 'chilly weather', 'entity': 'occasion'},
    ]},
    {'text': 'I want a net skirt for a summer party.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 12, 'value': 'net', 'entity': 'material'},
        {'start': 13, 'end': 18, 'value': 'skirt', 'entity': 'bottom_type'},
        {'start': 30, 'end': 35, 'value': 'party', 'entity': 'occasion'},
        {'start': 23, 'end': 29, 'value': 'summer', 'entity': 'weather'}
    ]},
    {'text': "I'm searching for a creap dress for an elegant look.", 'intent': 'get_attributes', 'entities': [
        {'start': 23, 'end': 28, 'value': 'creap', 'entity': 'material'},
        {'start': 29, 'end': 34, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 39, 'end': 51, 'value': 'elegant look', 'entity': 'style'},
    ]},
    {'text': 'Looking for a bubble skirt for a fun event.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 27, 'value': 'bubble', 'entity': 'material'},
        {'start': 28, 'end': 33, 'value': 'skirt', 'entity': 'bottom_type'},
        {'start': 38, 'end': 43, 'value': 'event', 'entity': 'occasion'},
    ]},
    {'text': 'I want a polyester top for a casual outing.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 18, 'value': 'polyester', 'entity': 'material'},
        {'start': 19, 'end': 22, 'value': 'top', 'entity': 'outfit_type'},
        {'start': 27, 'end': 33, 'value': 'casual', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a print dress for a casual outing.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 26, 'value': 'print', 'entity': 'style'},
        {'start': 27, 'end': 32, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 37, 'end': 44, 'value': 'casual', 'entity': 'occasion'},
    ]},
    {'text': 'I want a pleat skirt for a chic look.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 14, 'value': 'pleat', 'entity': 'style'},
        {'start': 15, 'end': 20, 'value': 'skirt', 'entity': 'outfit_type'},
        {'start': 25, 'end': 33, 'value': 'chic look', 'entity': 'style'},
    ]},
    {'text': 'Looking for a tiger print blouse for a bold style.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 31, 'value': 'tiger print', 'entity': 'style'},
        {'start': 32, 'end': 38, 'value': 'blouse', 'entity': 'outfit_type'},
        {'start': 43, 'end': 53, 'value': 'bold style', 'entity': 'style'},
    ]},
    {'text': 'I want a fleet dress for a special occasion.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 14, 'value': 'fleet', 'entity': 'style'},
        {'start': 15, 'end': 20, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 25, 'end': 32, 'value': 'special', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a lace top for an elegant look.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 25, 'value': 'lace', 'entity': 'style'},
        {'start': 26, 'end': 29, 'value': 'top', 'entity': 'outfit_type'},
        {'start': 34, 'end': 46, 'value': 'elegant look', 'entity': 'style'},
    ]},
    {'text': 'I want a stretch dress for a comfortable day.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 16, 'value': 'stretch', 'entity': 'style'},
        {'start': 17, 'end': 22, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 27, 'end': 43, 'value': 'comfortable day', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a checked coat for a cozy winter.', 'intent': 'get_attributes', 'entities': [
        {'start': 14, 'end': 21, 'value': 'checked', 'entity': 'style'},
        {'start': 22, 'end': 26, 'value': 'coat', 'entity': 'outfit_type'},
        {'start': 38, 'end': 44, 'value': 'winter', 'entity': 'weather'},
    ]},
    {'text': 'I want a polka dots dress for a retro vibe.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 19, 'value': 'polka dots', 'entity': 'style'},
        {'start': 20, 'end': 25, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 36, 'end': 40, 'value': 'vibe', 'entity': 'style'},
    ]},
    {'text': 'Looking for a loose blouse for a casual day.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 26, 'value': 'loose', 'entity': 'style'},
        {'start': 27, 'end': 32, 'value': 'blouse', 'entity': 'outfit_type'},
        {'start': 37, 'end': 45, 'value': 'casual', 'entity': 'occasion'},
    ]},
    {'text': 'I want a stone embellished dress for a glamorous look.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 27, 'value': 'stone embellished', 'entity': 'style'},
        {'start': 28, 'end': 33, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 38, 'end': 52, 'value': 'glamorous look', 'entity': 'style'},
    ]},
    {'text': 'Looking for a layered skirt for a trendy outfit.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 28, 'value': 'layered', 'entity': 'style'},
        {'start': 29, 'end': 34, 'value': 'skirt', 'entity': 'outfit_type'},
        {'start': 39, 'end': 51, 'value': 'trendy outfit', 'entity': 'style'},
    ]},
    {'text': 'I want a buff jacket for a cool day.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 13, 'value': 'buff', 'entity': 'style'},
        {'start': 14, 'end': 20, 'value': 'jacket', 'entity': 'outfit_type'},
        {'start': 25, 'end': 34, 'value': 'cool day', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a multi pattern dress for a lively event.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 31, 'value': 'multi pattern', 'entity': 'style'},
        {'start': 32, 'end': 37, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 49, 'end': 54, 'value': 'event', 'entity': 'occasion'},
    ]},
    {'text': 'I want a dress with vertical lines for a slimming effect.', 'intent': 'get_attributes', 'entities': [
        {'start': 24, 'end': 38, 'value': 'vertical lines', 'entity': 'style'},
        {'start': 39, 'end': 44, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 49, 'end': 65, 'value': 'slimming effect', 'entity': 'style'},
    ]},
    {'text': 'Looking for a feather trimmed skirt for a fancy occasion.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 36, 'value': 'feather trimmed', 'entity': 'style'},
        {'start': 37, 'end': 42, 'value': 'skirt', 'entity': 'outfit_type'},
        {'start': 53, 'end': 61, 'value': 'occasion', 'entity': 'occasion'},
    ]},
    {'text': 'I want a grand gown for a formal event.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 14, 'value': 'grand', 'entity': 'style'},
        {'start': 15, 'end': 19, 'value': 'gown', 'entity': 'outfit_type'},
        {'start': 24, 'end': 30, 'value': 'formal', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a simple blouse for everyday wear.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 27, 'value': 'simple', 'entity': 'style'},
        {'start': 28, 'end': 34, 'value': 'blouse', 'entity': 'outfit_type'},
        {'start': 39, 'end': 47, 'value': 'everyday', 'entity': 'occasion'},
    ]},
    {'text': 'I want a shiny dress for a night out.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 14, 'value': 'shiny', 'entity': 'style'},
        {'start': 15, 'end': 20, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 25, 'end': 33, 'value': 'night out', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a sketch patterned skirt for a creative event.', 'intent': 'get_attributes', 'entities': [
        {'start': 21, 'end': 36, 'value': 'sketch patterned', 'entity': 'style'},
        {'start': 37, 'end': 42, 'value': 'skirt', 'entity': 'outfit_type'},
        {'start': 56, 'end': 61, 'value': 'event', 'entity': 'occasion'},
    ]},
    {'text': 'I need a formal dress for an event in New York.', 'intent': 'get_attributes', 'entities': [
        {'start': 37, 'end': 45, 'value': 'New York', 'entity': 'region'},
        {'start': 9, 'end': 15, 'value': 'formal', 'entity': 'style'},
        {'start': 16, 'end': 21, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 28, 'end': 33, 'value': 'event', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a printed kimono for a Japan trip.', 'intent': 'get_attributes', 'entities': [
        {'start': 14, 'end': 21, 'value': 'printed', 'entity': 'style'},
        {'start': 22, 'end': 28, 'value': 'kimono', 'entity': 'outfit_type'},
        {'start': 35, 'end': 40, 'value': 'Japan', 'entity': 'region'},
        {'start': 41, 'end': 45, 'value': 'trip', 'entity': 'occasion'},
    ]},
    {'text': 'I want a casual shirt for an India vacation.', 'intent': 'get_attributes', 'entities': [
        {'start': 29, 'end': 34, 'value': 'India', 'entity': 'region'},
        {'start': 9, 'end': 15, 'value': 'casual', 'entity': 'style'},
        {'start': 16, 'end': 21, 'value': 'shirt', 'entity': 'outfit_type'},
        {'start': 35, 'end': 43, 'value': 'vacation', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a traditional qipao for a China trip.', 'intent': 'get_attributes', 'entities': [
        {'start': 14, 'end': 25, 'value': 'traditional', 'entity': 'style'},
        {'start': 26, 'end': 31, 'value': 'qipao', 'entity': 'outfit_type'},
        {'start': 38, 'end': 43, 'value': 'China', 'entity': 'region'},
        {'start': 44, 'end': 48, 'value': 'trip', 'entity': 'occasion'},
    ]},
    {'text': 'I need a cozy sweater for the Russia cold weather.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 13, 'value': 'cozy', 'entity': 'style'},
        {'start': 14, 'end': 21, 'value': 'sweater', 'entity': 'outfit_type'},
        {'start': 30, 'end': 36, 'value': 'Russia', 'entity': 'region'},
        {'start': 37, 'end': 41, 'value': 'cold', 'entity': 'weather'},
    ]},
    {'text': 'Looking for a breezy dress for a Bali vacation.', 'intent': 'get_attributes', 'entities': [
        {'start': 14, 'end': 20, 'value': 'breezy', 'entity': 'style'},
        {'start': 21, 'end': 26, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 33, 'end': 37, 'value': 'Bali', 'entity': 'region'},
        {'start': 38, 'end': 46, 'value': 'vacation', 'entity': 'occasion'},
    ]},
    {'text': 'I want a formal blazer for a business meeting in London.', 'intent': 'get_attributes', 'entities': [
        {'start': 49, 'end': 55, 'value': 'London', 'entity': 'region'},
        {'start': 9, 'end': 15, 'value': 'formal', 'entity': 'style'},
        {'start': 16, 'end': 22, 'value': 'blazer', 'entity': 'outfit_type'},
        {'start': 29, 'end': 45, 'value': 'business meeting', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a flowy skirt for an Australia vacation.', 'intent': 'get_attributes', 'entities': [
        {'start': 33, 'end': 42, 'value': 'Australia', 'entity': 'region'},
        {'start': 14, 'end': 19, 'value': 'flowy', 'entity': 'style'},
        {'start': 20, 'end': 25, 'value': 'skirt', 'entity': 'bottom_type'},
        {'start': 26, 'end': 35, 'value': 'Australia', 'entity': 'region'},
        {'start': 43, 'end': 51, 'value': 'vacation', 'entity': 'occasion'},
    ]},
    {'text': 'I want a formal suit for a London wedding.', 'intent': 'get_attributes', 'entities': [
        {'start': 27, 'end': 33, 'value': 'London', 'entity': 'region'},
        {'start': 9, 'end': 15, 'value': 'formal', 'entity': 'style'},
        {'start': 16, 'end': 20, 'value': 'suit', 'entity': 'outfit_type'},
        {'start': 34, 'end': 41, 'value': 'wedding', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a chic dress for a Paris vacation.', 'intent': 'get_attributes', 'entities': [
        {'start': 31, 'end': 36, 'value': 'Paris', 'entity': 'region'},
        {'start': 14, 'end': 18, 'value': 'chic', 'entity': 'style'},
        {'start': 19, 'end': 24, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 24, 'end': 30, 'value': 'Paris', 'entity': 'region'},
        {'start': 37, 'end': 45, 'value': 'vacation', 'entity': 'occasion'},
    ]},
    {'text': 'I need a traditional dress for a Russian winter.', 'intent': 'get_attributes', 'entities': [
        {'start': 33, 'end': 40, 'value': 'Russian', 'entity': 'region'},
        {'start': 9, 'end': 20, 'value': 'traditional', 'entity': 'style'},
        {'start': 21, 'end': 26, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 31, 'end': 38, 'value': 'Russian', 'entity': 'region'},
        {'start': 41, 'end': 47, 'value': 'winter', 'entity': 'weather'},
    ]},
    {'text': 'Looking for a modern outfit for a Japan vacation.', 'intent': 'get_attributes', 'entities': [
        {'start': 43, 'end': 48, 'value': 'Japan', 'entity': 'region'},
        {'start': 14, 'end': 20, 'value': 'modern', 'entity': 'style'},
        {'start': 21, 'end': 27, 'value': 'outfit', 'entity': 'outfit_type'},
        {'start': 34, 'end': 39, 'value': 'Japan', 'entity': 'region'},
        {'start': 40, 'end': 48, 'value': 'vacation', 'entity': 'occasion'},
    ]},
    {'text': 'I want a cozy sweater for the New York cold weather.', 'intent': 'get_attributes', 'entities': [
        {'start': 9, 'end': 13, 'value': 'cozy', 'entity': 'style'},
        {'start': 14, 'end': 21, 'value': 'sweater', 'entity': 'outfit_type'},
        {'start': 29, 'end': 37, 'value': 'New York', 'entity': 'region'},
        {'start': 38, 'end': 42, 'value': 'cold', 'entity': 'weather'},
    ]},
    {'text': 'Looking for a floral dress for a Chinese New Year celebration.', 'intent': 'get_attributes', 'entities': [
        {'start': 33, 'end': 40, 'value': 'Chinese', 'entity': 'region'},
        {'start': 14, 'end': 20, 'value': 'floral', 'entity': 'style'},
        {'start': 21, 'end': 26, 'value': 'dress', 'entity': 'outfit_type'},
        {'start': 26, 'end': 33, 'value': 'Chinese', 'entity': 'region'},
        {'start': 41, 'end': 61, 'value': 'New Year celebration', 'entity': 'occasion'},
    ]},
    {'text': 'I need a stylish blazer for a Tokyo business meeting.', 'intent': 'get_attributes', 'entities': [
        {'start': 30, 'end': 35, 'value': 'Tokyo', 'entity': 'region'},
        {'start': 9, 'end': 16, 'value': 'stylish', 'entity': 'style'},
        {'start': 17, 'end': 23, 'value': 'blazer', 'entity': 'outfit_type'},
        {'start': 36, 'end': 49, 'value': 'business meet', 'entity': 'occasion'},
    ]},
    {'text': 'Looking for a traditional qipao for a Beijing trip.', 'intent': 'get_attributes', 'entities': [
        {'start': 38, 'end': 45, 'value': 'Beijing', 'entity': 'region'},
        {'start': 14, 'end': 25, 'value': 'traditional', 'entity': 'style'},
        {'start': 26, 'end': 31, 'value': 'qipao', 'entity': 'outfit_type'},
        {'start': 31, 'end': 38, 'value': 'Beijing', 'entity': 'region'},
        {'start': 46, 'end': 50, 'value': 'trip', 'entity': 'occasion'},
    ]},
]




def train_spacy(data, iterations=30):
    TRAIN_DATA = data

    nlp = spacy.blank("en")  # start with a blank model

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for example in TRAIN_DATA:
        for entity in example["entities"]:
            ner.add_label(entity["entity"])

    losses = None
    optimizer = nlp.begin_training()

    for itn in range(iterations):
        print("Starting iteration " + str(itn))
        random.shuffle(TRAIN_DATA)
        losses = {}

        for example in TRAIN_DATA:
            doc = nlp.make_doc(example["text"])

            example_ents = [
                (entity["start"], entity["end"], entity["entity"])
                for entity in example["entities"]
            ]

            example_ents = list(set(example_ents))  # Remove duplicates
            example = Example.from_dict(doc, {"entities": example_ents})

            nlp.update([example], drop=0.2, losses=losses)

    print(losses)
    return nlp

if __name__ == "__main__":
    def func(sentence, word):
        start_index = sentence.find(word)
        end_index = start_index + len(word)
        return [start_index, end_index]

    for entry in TRAINING_DATA:
        for vals in entry["entities"]:
            correct_idx = func(entry["text"], vals["value"])
            vals["start"] = correct_idx[0]
            vals["end"] = correct_idx[1]
    # print(TRAINING_DATA)

    nlp = train_spacy(TRAINING_DATA)
    output_dir = "spacy_model_chatbot"
    nlp.to_disk(output_dir)
    print(nlp)

