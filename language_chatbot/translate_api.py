
from dotenv import load_dotenv, find_dotenv
import os
import requests


# loading the key from the .env file
env_path = find_dotenv() # automatic find
load_dotenv(env_path)
key = os.getenv('DEEPL_AUTH_KEY')

# creating variables
url = 'https://api-free.deepl.com/v2/translate'
text = 'hello, world'
tar_language = 'DE'


def translate(text, tar_language, url):
    """
    Translates to the chosen language
    Please provide the following as input:
    (1) text you want to translate,
    (2) targer language e.g. DE,
    (3) URL (by default, it is DEEPL API)
    Output: translated text
    """

    response = requests.get(url, params={'auth_key': key, 'text': text, 'target_lang': tar_language}).json()

    return response['translations'][0]['text']



if __name__ == '__main__':
    print(translate(text, tar_language, url))
