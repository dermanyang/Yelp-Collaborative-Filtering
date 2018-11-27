import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sys
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True
#
#   PATH: "train_reviews.csv/train_reviews.csv"
#

train_reviews_file_name = 'train_reviews.csv/train_reviews.csv'
business_file_name = 'business.csv/business.csv'
users_file_name = 'users.csv/users.csv'
# test_queries_file_name = 'test_queries.csv/test_queries.csv'


N = 0
with open(train_reviews_file_name, newline='', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        try:
            N += 1
        except IOError:
            pass

##                     ##
##  Rename businessess ##
##                     ##
business_map = {}
count = 0
with open(business_file_name, newline='', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        business_id = row[41]
        if not business_id in business_map:
            business_map[business_id] = count
            count += 1

##               ##
##  Rename users ##
##               ##
users_map = {}
count = 0
with open(users_file_name, newline='', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        user_id = row[20]
        if not user_id in users_map:
            users_map[user_id] = count
            count += 1

# print('TEST', users_map['VDh1vjzpNUJH6HfcjH8g7Q'])

with open(train_reviews_file_name, newline='', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    # row_count = sum(1 for row in csv_reader)  # fileObject is your csv.reader

    # x_train holds information about the user and business
    x_train = np.empty((N -1, 2), dtype="S40")


    # y_train is the stars
    y_train = np.zeros((N-1, 1))

    for i, row in enumerate(csv_reader):
        business_id = row[0]
        # print(row[5],type(row[5]))
        stars = float(row[5])
        user_id = row[8]
        x_train[i][0] = business_map[business_id]
        x_train[i][1] = users_map[user_id]
        y_train[i] = stars


print(x_train) # works
print(y_train)


# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(solver='lbfgs', multi_class='multinomial')


# fit the model according to the given training data
logisticRegr.fit(x_train, y_train)

# # x_test holds information about the user and business
# x_test = []t
#
predictions = logisticRegr.predict(x_train)
for i in range(len(predictions)):
    print("PREDICT: ", predictions[i], "\nACTUAL: ", y_train[i])
