from controller_our_case import train_MVCNN
from test_our_case import test_MVCNN

"""Number of different ModelNet40 from which the instances are taken from"""
NUMBER_CLASSES = 26

"""Number of instances from each class"""
NUMBER_INSTANCES = 1

"""Number of total objects used: NUMBER_CLASSES * NUMBER_INSTANCES"""

print("Total number of different objects: " + str(NUMBER_CLASSES*NUMBER_INSTANCES))

case_description = str(NUMBER_CLASSES) + "_" + str(NUMBER_INSTANCES)
train_MVCNN(case_description)

test_description = case_description
test_MVCNN(test_description)