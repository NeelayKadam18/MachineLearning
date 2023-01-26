# Always encode string with integer

# Import required library
from sklearn import tree

# Load the dataset
Features = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],[35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],
            [96,"Smooth"],[43,"Rough"],[110,"Smooth"],[35,"Rough"],[95,"Smooth"]]
Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket","Tennis","Tennis","Tennis","Cricket","Tennis","Cricket","Tennis","Cricket"]

# Decide the ML Algorithm
obj = tree.DecisionTreeClassifier()

# Perform the training of model
obj = obj.fit(Features,Labels)          # paramters - 1.List of Features 2. List of Labels

# Perform the testing
print(obj.predict([[97,"Smooth"]]))


