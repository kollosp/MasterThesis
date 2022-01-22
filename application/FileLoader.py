import numpy as np
from csv import reader

class FileLoader:

    @staticmethod
    def load_from_csv(file_path):
        data = []
        headers = []
        location = []
        # open file in read mode
        with open(file_path, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Iterate over each row in the csv using reader object
            index = 0
            for row in csv_reader:
                if index == 1:
                    data.append(row[3:])
                elif index > 2:
                    # row variable is a list that represents a row in csv
                    if len(row) >= 3:
                        #name of installation
                        headers.append(row[0])
                        #location latitude and longitude
                        location.append([row[1], row[2]])
                        data.append(row[3:])
                index += 1



        return headers, np.array(location).astype(np.float), np.array(data).astype(np.float)