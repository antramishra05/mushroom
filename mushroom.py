import sklearn
from sklearn.model_selection import train_test_split

X_train = y_train = []
X_test = y_test = []
X_cross_valid = y_cross_valid = []
feature_name = list()


def map_feature_index_to_name():
    global feature_name
    feature_name = [
        'cap-shape' ,
        'cap-surface',
        'cap-color',
        'bruises',
        'odor',
        'gill-attachment',
        'gill-spacing',
        'gill-size',
        'gill-color',
        'stalk-shape',
        'stalk-surface-above-ring',
        'stalk-surface-below-ring',
        'veil-type','veil-color',
        'ring-number',
        'ring-number',
        'ring-type',
        'spore-print-color',
        'population',
        'habitat'
    ]


def split_dataset():
    global X_train,X_test,X_cross_valid,y_train,y_test,y_cross_valid
    with open("dataset.txt") as f:
        dataset = f.readlines()
    dataset = [x.strip() for x in dataset]


    output_array = []
    feature_array = []

    for line in dataset:
        tmp = line.split(",")
        output_array.append(tmp[0])
        feature_array.append(tmp[1:])

    X_train, X_test, y_train, y_test = train_test_split(feature_array, output_array, test_size=0.20, random_state=42)
    X_train, X_cross_valid, y_train, y_cross_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print("Done splitting dataset")

def accuracy(y_test,y_prediction):
    i = 0
    count = 0
    for p in y_prediction:
        actual_output = y_test[i]
        if(p == actual_output):
            count += 1
        i += 1
    accuracy = (count*100)/y_test
    print(accuracy)


def calc_probabilty_of_features(feature_vector,output_vector):
    feature_prob = list()
    prob_poisonous = 0 # probability that a mushroom is poisonous
    prob_feature = list() # probability of features
    i = 0
    total_poisonous = 0

    for y in output_vector:
      if(y == 'p'):
          total_poisonous += 1

    prob_poisonous = total_poisonous/len(output_vector)
    features_value = dict()
    i = 0
    for features_of_example in feature_vector:
        index = 1
        for f in features_of_example:
            if index not in features_value:
                features_value[index] = dict()

            feature_val_hash = features_value[index]
            if(f not in feature_val_hash):
                feature_val_hash[f] = 0
            feature_val_hash[f] += 1
            index += 1

    return features_value


map_feature_index_to_name()
split_dataset()
calc_probabilty_of_features(X_train,y_train)