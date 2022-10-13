import pandas as pd

# read in data
iris_data = pd.read_csv('iris_data/IRIS.csv')
#print(iris_data)

# shuffle data
shuffled_data = iris_data.sample(frac = 1)
#print(shuffled_data)

# sample training and test sets
train_set = shuffled_data.iloc[:100,:]
test_set = shuffled_data.iloc[100:,:]
#print(train_set)
# print(len(test_set))

# closeness = x, if x is small, examples a and b are close
# closeness = diff1^2 + diff2^2 + diff3^2 + diff4^2
# diff
def compare_prediction(prediction, example):
    # get species from test example
    return example[4] == prediction

def k_nearest(k, train_set, test_set):
    """
        K-nearest neighbors
        1. get closeness between test example and every training example
        2. get k closest training examples (for this test example)
        3. get most frequently occuring species out of k closest training examples, classify test example as that 
    """
    # 1.
    num_correct = 0
    total_num = 0
    for test_ex in test_set.iterrows():
        test_ex = test_ex[1]
        closeness = {}
        i=0
        for train_ex in train_set.iterrows():
            train_ex = train_ex[1]
            # test_ex[0] = sepal_length
            # test_ex[1] = sepal_width etc.
            # closeness = diff_between_sepal_length^2 + diff_between_sepal_width^2 +...etc
            # diff_between_sepal_length = test_ex[0] - train_ex[0]
            closeness[((test_ex[0] - train_ex[0])**2 + (test_ex[1] - train_ex[1])**2 + (test_ex[2] - train_ex[2])**2 + (test_ex[3] - train_ex[3])**2)] = i
            i += 1
            # verify closeness

        # 2. get k closest examples
        sorted_keys = sorted(closeness.keys())
        k_closest_exs = []
        for j in range(k):
            k_closest_exs.append(closeness[sorted_keys[j]])


        # 3. get most frequently occuring species out of k closest training examples, classify test example as that 
        occurances = {}
        for ex_idx in k_closest_exs:
            # example_index = the row index of the example we want in "train_set"
            # get that example's species
            species_name = train_set.iloc[ex_idx]['species']

            # add that to occurances
            # dict[key] = value
            # occurances[species_name] =  num_of_occurances_so_far_of_that_species
            if species_name not in occurances.keys():
                occurances[species_name] = 1
            occurances[species_name] += 1
        
        max_value = max(occurances.values())
        for key in occurances.keys():
            if occurances[key] == max_value:
                prediction = key
                break

        # check prediction accuracy
        if (compare_prediction(prediction, test_ex)):
            num_correct += 1
        total_num += 1
    print('prediction accuracy = ', num_correct/total_num)

k_nearest(11, train_set, test_set)