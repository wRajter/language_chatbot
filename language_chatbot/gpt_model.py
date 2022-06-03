from fnmatch import translate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from language_chatbot.translate_api import translate
from language_chatbot.detect import detect_language


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


def chat(new_user_input, user_language):

    #Checking if user specified a language
    if  user_language == 'no lang':
        language_detect = detect_language(new_user_input)

    else:
        language_detect = user_language

    if len(language_detect) > 8:
        return language_detect
    else:
        # user's input and response are traslated to English
        translated_input = translate(new_user_input, 'EN')

        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(translated_input + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([new_user_input_ids], dim=-1)

        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        bot_decoded_ans = format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

        #TODO: translate the answer from English to desired language
        translated_output = translate(bot_decoded_ans, language_detect)

        return translated_output


if __name__ == '__main__':
    print('>>>>>>type quit to quit the program<<<<<<<')
    print(chat("why do I need to do my homework?", "de"))
