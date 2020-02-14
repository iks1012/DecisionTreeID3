class Node:
    def __init__(self, attrib_index, comparator_threshold, children, answer, data):
        self.attrib_index = attrib_index
        self.comparator_threshold = comparator_threshold
        self.children = children
        self.label_answer = answer
        self.associated_data = data
        self.left_child = -1
        self.right_child = -1

    def __init__(self, data, labels):
        #print("new Node()")
        self.attrib_index = -1
        self.is_leaf = True
        self.comparator_threshold = -1
        self.children = []
        self.children_split_values = []
        self.label_answer = '-1'
        self.associated_data = data
        self.associated_labels = labels
        self.left_child = -1
        self.right_child = -1
        self.attrib_isDiscrete = False
        
    