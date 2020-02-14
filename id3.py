import os, sys, errno, csv, argparse, re, string
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import pylab
import numpy as np # only need this for some array stuff (coming up with array for axis ticks for plot)


from node import Node # class that I made (node.py)
from math import log

column_labels = []

def main():


    print("CSE 353 Programming Part of HW2 (Spring 2019):")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', help='Please indicate path to the dataset with filename', default = '<NOT_SELECTED>', required = True)
    args = vars(parser.parse_args())

    dataset_file_path = validate_args(args)


    x_names, y_names, x_test, y_test, x_train, y_train = compile_data_array(dataset_file_path, testPortion = 0.1, trainPortion = 0.9) # implement this next
    
    column_labels = x_names
    
    print("Populating Decision Tree from Training data ... ")
    terminal_root = id3(x_train, y_train, range(len(x_train[0])))

    print("Done! - acquired accuracy values")


    print("Calculating Accuracy for depth ... ")
    max_depth = get_depth_of_tree(terminal_root)
    train_acc = compile_accuracy_array(x_train, y_train, terminal_root, max_depth)
    test_acc = compile_accuracy_array(x_test, y_test, terminal_root, max_depth)

    print("Final Training Accuracy : %.3f " %  calculate_accuracy(x_train, y_train, terminal_root, max_depth+1) )
    print("Final Test Accuracy : %.3f " %  calculate_accuracy(x_test, y_test, terminal_root, max_depth+1) )
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pylab.grid(True, lw = 1, ls = '--', c = '.75')
    pylab.plot(range(len(train_acc)), train_acc, 'r', label = 'Training') 
    pylab.plot(range(len(test_acc)), test_acc, 'b', label = 'Test')
    pylab.legend(loc='upper left')
    pylab.ylim(40, 100)

    plt.xticks(np.arange(0, 9, 1))
    plt.yticks(np.arange(40, 100, 5))

    pylab.ylabel('Accuracy')
    pylab.xlabel('Depth')
    pylab.show()

    

def get_depth_of_tree(root):
    if root.is_leaf:
        return 1
    
    if root.attrib_isDiscrete:
        # iterate through children[]
        depths = []
        for child in root.children:
            depths.append(get_depth_of_tree(child))
        return max(depths) + 1
    else:
        left_depth = get_depth_of_tree(root.left_child)

        right_depth = get_depth_of_tree(root.right_child)

        return max(left_depth, right_depth) + 1


def label_to_number(label):
    if label == '1':
        return 1
    if label == '0':
        return 0
    return label


def compile_accuracy_array(data, labels, root, max_depth):
    acc = []
    for i in range(max_depth):
        acc.append(calculate_accuracy(data, labels, root, i))

    return acc

def calculate_accuracy(data, labels, root, max_depth):

    num_correct = 0
    num_total = 0
    for i in range(len(data)):
        answer = predict_survivability(data[i], root, 0, max_depth)
        if answer == label_to_number(labels[i]):
            num_correct += 1
        num_total += 1
    
    return num_correct * 100 / num_total


#
# returns 1 if survived
# returns 0 if not survived
#
def predict_survivability(tup, root, depth, max_depth):
    answer = '-1'
    if root.is_leaf == True:
        if root.label_answer == '-1':
            answer = label_to_number(get_majority_label(root.associated_labels))
        else:
            answer = label_to_number(root.label_answer)
        return answer


    if depth+1 >= max_depth:
        return label_to_number(get_majority_label(root.associated_labels))
    
    a_index = root.attrib_index

    if get_variable_type(a_index) == 0: # discrete
        path_index = -1
        for i in range(len(root.children_split_values)):
            if root.children_split_values[i] == tup[a_index]:
                path_index = i
        answer = label_to_number(predict_survivability(tup, root.children[path_index], depth+1, max_depth))
    elif get_variable_type(a_index) == 1: # continuous
        if tup[a_index] < root.comparator_threshold:
            answer = label_to_number(predict_survivability(tup, root.left_child, depth+1, max_depth)) # go to left child
        else:
            answer = label_to_number(predict_survivability(tup, root.right_child, depth+1, max_depth)) # go to right child

    return answer

def sort_data_by_index_increasing(data, index):
    for i in range(len(data)):
        for j in range(len(data)-i-1):
            if data[j][index] > data[j+1][index]:
                data[j][index], data[j+1][index] = data[j+1][index], data[j][index]
    return data

def get_column_name(idx):
    global column_labels
    return column_labels[idx]


'''
-1: invalid index inputted
0 : discrete variable
1 : continuous variable
''' 
def get_variable_type(dataset_index):
    switcher = {
        0 : 0,
        1 : 0,
        2 : 1,
        3 : 0,
        4 : 0,
        5 : 1,
        6 : 1,
        7 : 0
    }
    return switcher.get(dataset_index, -1)

def c_function(a):
    if (a == 1.0):
        a -= 0.0000000001 # just another edge case
    if (a == 0.0):
        a += 0.0000000001 # just another edge case
    c = -1*a*log(a,2) - (1-a)*log(1-a,2)
    return c





def computeIGForColumn_continuous(data, index, labels):
    # find the max and min in the index
    max_val = -1
    min_val = 99999
    x = 3
    for tup in data:
        if tup[index] > max_val:
            max_val = tup[index]
        if tup[index] < min_val:
            min_val = tup[index]
    
    
    # Compute the interval for the thresholds to test
    number_of_steps = 10
    interval = ( max_val - min_val ) / number_of_steps

    # Get the max of each threshold in the range

    data = sort_data_by_index_increasing(data, index)




    max_IG_threshold = 0.0
    current_threshold = min_val
    for i in range(len(data)-1):
        if labels[i] != labels[i+1]:
            current_threshold = data[i][index] + (data[i+1][index] - data[i][index])/2
            #print(current_threshold)
            temp = computeIGForSpecificThreshold(data, index, labels, current_threshold)
            if temp > max_IG_threshold:
                max_IG_threshold = current_threshold
    # return the max threshold for this index bc it is continous
    

    return max_IG_threshold

def computeIGForSpecificThreshold(data, index, labels, threshold):

    # IG(Y|t) = H(Y) - H(Y|t) such that t is the threshold
    # H(Y|t) = P(X<t)*H(Y|X<t) + P(X>=t)*H(Y|X>=t)
    
    # Compute H(Y)
    occurances_yis1 = 0
    occurances_yis1_xlessthent = 0
    occurances_yis1_xgreaterthent = 0
    occurances_xlessthent = 0
    occurances_xgreaterthent = 0

    for i in range(len(data)):
        if labels[i] == '1':
            occurances_yis1 += 1
            if data[i][index] < threshold:
                occurances_yis1_xlessthent += 1
            if data[i][index] >= threshold:
                occurances_yis1_xgreaterthent += 1
        if labels[i] != '1':
            if data[i][index] < threshold:
                occurances_xlessthent += 1
            if data[i][index] >= threshold:
                occurances_xgreaterthent += 1
            
    p_xigt = occurances_xgreaterthent / len(data)
    p_xilt = occurances_xlessthent / len(data)

    p_xigt_yis1 = occurances_yis1_xgreaterthent / occurances_yis1

    p_xilt_yis1 = occurances_yis1_xlessthent / occurances_yis1

    p_yis1 = occurances_yis1 / len(data)

    return c_function(p_yis1) - (p_xilt * c_function(p_xilt_yis1) + p_xigt * c_function(p_xigt_yis1))


def computeIGForColumn(data, index, labels):
    num_ones = 0
    for value in labels:
        if value == '1':
            num_ones += 1
    
    p_yis1 = num_ones / len(labels)

    c_p_yis1 = c_function(p_yis1)


    value_map, value_map_given_yis1 = get_num_occurances(data, index, labels)

    # Compute the conditional entropy for each unique value in the attribute
    running_sum = 0
    
    for key in value_map.keys(): 

        p_xiskey = value_map[key] / len(data)
        p_xiskey_given_yis1 = 0
        if key not in value_map_given_yis1:
            p_xiskey_given_yis1 = 0.0
        else:
            p_xiskey_given_yis1 = value_map_given_yis1[key] / value_map[key]
        running_sum += (p_xiskey*c_function(p_xiskey_given_yis1))

    return c_function(p_yis1) - running_sum

def get_num_occurances(data, index, labels):

    # hash map of values
    value_map = {} # used for computing P(x) for all unique inputs
    value_map_given_yis1 = {} # used for computing P(y = 1 | x) for all unique inputs

    # get num occurances of each unique value
    for i in range(len(data)):
        tup = data[i]
        # get num occurances
        if tup[index] not in value_map:
            value_map[tup[index]] = 1
        else:
            value_map[tup[index]] = value_map[tup[index]] + 1

        # get the times it is also 1
        if labels[i] == '1':
            if tup[index] not in value_map_given_yis1:
                value_map_given_yis1[tup[index]] = 1
            else:
                value_map_given_yis1[tup[index]] = value_map_given_yis1[tup[index]] + 1

    return value_map, value_map_given_yis1


def id3(data, labels, index_list):

    
    root = Node(data, labels)
    root = id3_basecases(root, data, labels, index_list)
    if root.label_answer != '-1': # if the answer was assigned
        return root

    # split on best feature, separate the data by that feature
    IGs = []
    max_ig = -1
    max_ig_index = -1
    for i in index_list:
        temp = -1
        if get_variable_type(i) == 0:
            temp = (computeIGForColumn(data, i, labels))
        else:
            temp = (computeIGForSpecificThreshold(data, i, labels, computeIGForColumn_continuous(data, i, labels)))
        if temp > max_ig:
            max_ig = temp
            max_ig_index = i
    
    
    root.attrib_index = max_ig_index
    index_of_feature_split = index_list.index(max_ig_index)
    new_indices = []

    for i in index_list:
        if i != max_ig_index:
            new_indices.append(i)
    
    
    index_list = new_indices

    start_val = data[0][max_ig_index]
    is_different = False
    for tup in data:
        if tup[max_ig_index] != start_val:
            is_different = True
    if not is_different:
        root.label_answer = get_majority_label(labels)
        return root


    if get_variable_type(max_ig_index) == 1: # it is continuous , split greater then/lessthen threshold
        threshold = computeIGForColumn_continuous(data, max_ig_index, labels)
        root.comparator_threshold = threshold

        root.is_leaf = False
        # Split dataset 
        left_data = []
        right_data = []

        left_labels = []
        right_labels = []
        for i in range(len(data)):
            tup = data[i]

            if tup[max_ig_index] >= threshold:
                right_data.append(tup)
                right_labels.append(labels[i])
            else:
                left_data.append(tup)
                left_labels.append(labels[i])
        
        root.left_child = id3(left_data,left_labels, index_list)
        root.right_child = id3(right_data, right_labels, index_list)
    elif get_variable_type(max_ig_index) == 0: # it is discrete, create child on each of the unique options
        value_map, value_map_given_yis1 = get_num_occurances(data, max_ig_index, labels)
        #print("discrete split")
        root.attrib_isDiscrete = True
        root.is_leaf = False
        for key in value_map.keys():
            temp_dataset = []
            temp_labelset = []
            for i in range(len(data)):
                if data[i][max_ig_index] == key:
                    temp_dataset.append(data[i])
                    temp_labelset.append(labels[i])
            root.children_split_values.append(key)
            root.children.append(id3(temp_dataset, temp_labelset, index_list))
    # the data set splits are going to go with each child

    return root

def id3_basecases(root, data, labels, index_list):
    # Base cases
    is_one = True
    num_ones = 0
    is_zero = True
    num_zeroes = 0
    for i in range(len(labels)):
        if labels[i] == '1':
            is_zero = False
            num_ones += 1
        if labels[i] == '0':
            is_one = False
            num_zeroes += 0
    
    # Base case one
    if len(index_list) == 0:
        root.label_answer = get_majority_label(labels)
        return root
        
    # Base Case two
    if is_zero:
        root.label_answer = '0'
        return root
    
    # Base Case three
    if is_one:
        root.label_answer = '1'
        return root

    return root

def get_majority_label(labels):
    num_ones = 0
    num_zeroes = 0

    for num in labels:
        if num == '1':
            num_ones += 1
        else:
            num_zeroes += 1
    
    if num_ones >= num_zeroes:
        return '1'
    else:
        return '0'


def embarkGetIndex(ch):

    switcher = {
        'S' : 0,
        'Q' : 1,
        'C' : 2
    }
    return switcher.get(ch, -1)

def embarkGetChar(number):
    switcher = {
        0 : 'S',
        1 : 'Q',
        2 : 'C'
    }
    return switcher.get(number, -1)


def string_to_int(the_str):
    answer = 0
    for char in the_str:
        answer += ord(char)
    return answer

age_sum = 0
init_call = 0

def preprocess_row(row):
    global init_call
    global age_sum
    new_row = []
    
    


    if init_call == 0:
        # this is the first call so its all strings atm for the headers of each column
        new_row.append(row[2])
        new_row.append(row[4])
        new_row.append(row[5])
        new_row.append(row[6])
        new_row.append(row[7])
        new_row.append(row[8])
        new_row.append(row[9])
        new_row.append(row[11])
        init_call = 1
        return new_row

    # [0]: PClass
    new_row.append(int(row[2]))

    # [1]: Sex
    if row[4] == 'male':
        new_row.append(0) # male
    else:
        new_row.append(1) # female

    
    # [2]: Age
    if row[5] == '':
        new_row.append(-1)
    else:
        age = float(row[5])
        age_sum += age
        new_row.append(age)

    # [3]: SibSp
    new_row.append(int(row[6]))

    # [4]: Parch
    new_row.append(int(row[7]))


    # [5]: Ticket Num, basically take all the numbers from this and typecast to int

    if any(char.isdigit() for char in row[8]):
        new_row.append(int(re.sub(r'[^0-9]+','', row[8], re.I)))
    else:
        new_row.append(string_to_int(row[8]))

    # [6]: Fare
    new_row.append(float(row[9]))

    # [7]: Embarked
    new_row.append(embarkGetIndex(row[11]))

    return new_row

# the holes are -1
def updateHolesAtIndex(input_vals, index, value):
    for tup in input_vals:
        if tup[index] == -1:
            tup[index] = value

    return input_vals




def compile_data_array(file_path, testPortion, trainPortion):

    if (testPortion + trainPortion - 1.0 > 1e-4):
        print("Invalid Train and Test Ratio, must add up to one")
    dataset_file = open(file_path, 'r')
    dataset_csv = csv.reader(dataset_file, delimiter = ',')
    dataset = []
    x = []
    y = []

    prediction_index = 1
    
    for row in dataset_csv:
        preprocessed_row = preprocess_row(row)
        dataset.append(row)
        x.append((preprocessed_row))
        if row[prediction_index] == 'Survived': # Exclude the Header
            y += '?'
        else:
            y += row[prediction_index]

    global age_sum
    avg_age = age_sum / (len(x)-1) # -1 for the column labels
    x = updateHolesAtIndex(x, 2, avg_age)

    y_names = y[0]
    x_names = x[0]

    dataset_threshold_value = int(trainPortion * (len(x)-1))

    x_train = x[1:dataset_threshold_value]
    x_test = x[dataset_threshold_value+1:]

    y_train = y[1:dataset_threshold_value]
    y_test = y[dataset_threshold_value+1:]

    

    return x_names, y_names, x_test, y_test, x_train, y_train





def validate_args(args):
    dataset_file_path = args['dataset']

    if not valid_pathname(dataset_file_path):
        print("Please input a valid file name and destination, the file you entered does not exist at the moment")
        exit(-1)
    

    return dataset_file_path


def valid_pathname(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    try:
        if not isinstance(pathname, str) or not pathname:
            return False
        _, pathname = os.path.splitdrive(pathname)
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)

        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.errno == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    except TypeError as exc:
        return False
    else:
        return True


if __name__ == '__main__':
    main()