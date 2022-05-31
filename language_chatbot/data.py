import os
import pandas as pd
import yaml
import os


def getting_yaml_data(path_to_input_file='../raw_data/small conversational bot'):
    '''
    takes a yaml format input and returns pandas dataframe
    - please, specify the path to the input file,directory
    '''

    # as this dataset consits from multiple files, we will store all the file names and open them later one by one
    files = os.listdir(path_to_input_file) # path to the directory with all the YAML files


    # creating a dictionary where we will store data form all the files
    data_dic = {
        'tag': [],
        'patterns': [],
        'responses': []
    }

    # opening the YAML files one by one and storing the content into the dictionary
    for file in files:
        if 'Zone.Identifier' not in file:
            with open(f'{path_to_input_file}/{file}', "r") as stream:
                data = yaml.safe_load(stream)

                for record in data['conversations']:
                    data_dic['tag'].append(data['categories'][0])
                    data_dic['patterns'].append(record[0])
                    data_dic['responses'].append(record[1])

    # creating a pandas dataframe form the dictionary
    return pd.DataFrame.from_dict(data_dic)

if __name__ == '__main__':
    output = getting_yaml_data()
    print(output)
