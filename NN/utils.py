
import pickle
def writeto(object, filename):
    with open(filename, "wb") as file:  # Use 'wb' for writing binary
        pickle.dump(object, file)