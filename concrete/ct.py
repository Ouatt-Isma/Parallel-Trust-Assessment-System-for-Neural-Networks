# import numpy as np

# class Person:
#     def __init__(self, name, age, height):
#         self.name = name
#         self.age = age
#         self.height = height
    
#     def __str__(self):
#         return f"Name: {self.name}, Age: {self.age}, Height: {self.height}"
#     def __array__(self,dtype):
#         print('_AER_')
#         return np.array((self.name, self.age, self.height), dtype=dtype)
#     # Define magic methods to support addition
#     def __add__(self, other):
#         if isinstance(other, Person):
#             new_name = f"{self.name} & {other.name}"
#             new_age = self.age + other.age
#             new_height = self.height + other.height
#             return Person(new_name, new_age, new_height)
#         else:
#             raise TypeError("Unsupported operand type(s) for +: 'Person' and '{}'".format(type(other)))

#     # Define magic methods to support subtraction
#     def __sub__(self, other):
#         if isinstance(other, Person):
#             new_name = f"{self.name} without {other.name}"
#             new_age = self.age - other.age
#             new_height = self.height - other.height
#             return Person(new_name, new_age, new_height)
#         else:
#             raise TypeError("Unsupported operand type(s) for -: 'Person' and '{}'".format(type(other)))

#     # Define magic method to support dot product
#     def __matmul__(self, other):
#         if isinstance(other, Person):
#             return self.age * other.age + self.height * other.height
#         else:
#             raise TypeError("Unsupported operand type(s) for @: 'Person' and '{}'".format(type(other)))

#     # Define __array_function__ to enable numpy array functions
#     def __array_function__(self, func, types, args, kwargs):
#         if func is np.dot:
#             if len(args) == 2 and all(isinstance(arg, np.ndarray) for arg in args):
#                 return np.array([self @ person for person in args[1]]).astype(float)
#             else:
#                 return NotImplemented
#         else:
#             return NotImplemented

# # Create a NumPy array with the custom dtype
# person_dtype = np.dtype([('name', 'U50'), ('age', np.int32), ('height', np.float64)])
# people_array1 = np.array([Person('Alice', 30, 5.6), Person('Bob', 25, 6.0)], dtype=person_dtype)
# people_array2 = np.array([Person('Charlie', 35, 6.2), Person('David', 40, 5.8)], dtype=person_dtype)

# # Test np.dot
# result = np.dot(people_array1, people_array2)
# print(result)  # Output: [774. 814.]


import numpy as np

class CustomDataType:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __matmul__(self, other):
        if not isinstance(other, CustomDataType):
            raise ValueError("Operand must be an instance of CustomDataType")
        
        # Define how the dot product is computed. This is just an example.
        return self.a * other.a + self.b * other.b + self.c * other.c

    def __repr__(self):
        return f"CustomDataType(a={self.a}, b={self.b}, c={self.c})"

# Example usage:
element1 = CustomDataType(1, 2, 3)
element2 = CustomDataType(4, 5, 6)

# Using the @ operator for the dot product
result = element1 @ element2
print("Dot product:", result)

# For arrays of these custom objects, you'll need to manually handle operations
arr1 = np.array([CustomDataType(1, 2, 3), CustomDataType(4, 5, 6)])
arr2 = np.array([CustomDataType(7, 8, 9), CustomDataType(10, 11, 12)])

# Manually computing dot product for arrays of custom objects
dot_products = np.array([a @ b for a, b in zip(arr1, arr2)])
print("Dot products for arrays:", dot_products)
