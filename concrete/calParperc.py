import json
import numpy as np
from TrustOpinion import TrustOpinion
from ArrayTO import ArrayTO
from sklearn.model_selection import train_test_split

def std_loss(y, y_prime):
    return (y-y_prime)**2

def parse_training_log_file(log_file):
        f =open(log_file)
        data = json.load(f)
        return data 

class LinearRegression:
    def __init__(self, input_dim, hist, epsilon_low=0.1, epsilon_up = 1, init="vacuous"): 
        """
        init:
            - vacuous
            - trust
            - distrust
        """
        # hist: list of couple x,y_prime,y during the training.
        # self.weights = np.random.randn(input_dim + 1)  # +1 for the bias term
        # print("OK")
        # self.weights = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, 1)))

        self.weights = ArrayTO(TrustOpinion.fill(shape=(input_dim+1, 1), method=init))
        self.epsilon_low=epsilon_low
        self.epsilon_up = epsilon_up
        self.hist = hist 
        # print("OK")

    def forward_pass(self, X: ArrayTO):
        # Add bias term to input
        # one = np.ones((X.shape[0], 1))
        # print(X[0][0])
        one = TrustOpinion.fill(shape = (X.value.shape[0], 1), method="one") 
        X_with_bias = ArrayTO(np.c_[X.value, one])
        # print(X_with_bias[0][0])
        # Compute the dot product
        # print(X_with_bias.dtype)
        # print(self.weights.dtype)
        # print("W")
        # print(self.weights)
        # print("END W")
        # print("X_bia")
        # print(X_with_bias)
        # print("ENDX_bia")
        return ArrayTO.dot(X_with_bias, self.weights)
        # return X_with_bias@self.weights

    def backward_pass(self, X: ArrayTO, y: ArrayTO, learning_rate: TrustOpinion, loss):
        y_batch = y.fuse()
        # Add bias term to input
        one = TrustOpinion.fill(shape = (X.value.shape[0], 1)) 
        # X_with_bias = ArrayTO(np.c_[X.value, one])
        # y_prime = self.forward_pass(X)

        # # errors = predictions - y
        # errors = y 

        
        # gradients = ArrayTO.dot(X_with_bias.T, errors)
        # # gradients = errors 

        # self.weights = self.weights.update(learning_rate * gradients)
        opinion_theta_given_y = ArrayTO.theta_given_y(loss, self.epsilon_low)
        opinion_theta_given_not_y = ArrayTO.theta_given_not_y(loss, self.epsilon_up)
        # print("here")
        # print(y.value)
        # print(np.shape(y.value))
        # raise NotImplementedError
        opinion_theta_y = ArrayTO.op_theta_y(opinion_theta_given_y, opinion_theta_given_not_y, y_batch)


        opinion_theta = ArrayTO.op_theta(self.weights, opinion_theta_y)

        self.weights = opinion_theta.update(opinion_theta, learning_rate)


    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        mses = []
        # print(self.weights)
        for epoch in range(1, epochs+1):
            self.backward_pass(X_train, y_train, learning_rate, loss=np.array(self.hist["epochs"][epoch-1]["loss"]))

            # predictions = self.forward_pass(X_train)
            # print(f"Epoch {epoch}: MSE ")
            # print(self.weights[0][0].projected_prob())
                
        return mses 

def eval(log_file):
    to_eval = ["vacuous", "distrust", "trust"]
    # for weights_init in to_eval:
    weights_init = "vacuous"
    for x_op in to_eval:
        for y_op in to_eval:
            eval_one(log_file, x_op, y_op, weights_init)
        break
         
            
def eval_one(log_file, x_op="vacuous", y_op="vacuous", weights_init="vacuous"):
    print(f"weights Init => {weights_init} \n X => {x_op}\n  y => {y_op}\n")
    data = parse_training_log_file(log_file)
    # n_data = 10
    # d = 10
    n_data, d = (20640, 8)
    X = TrustOpinion.fill(shape = (n_data, d), method=x_op)
    y = TrustOpinion.fill(shape = (n_data, 1), method=y_op)
     # Split data into train and test sets
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = ArrayTO(X_train), ArrayTO(X_test), ArrayTO(y_train), ArrayTO(y_test)
    
    # Create and train the linear regression model
    linear_regression = LinearRegression(input_dim=X_train.value.shape[1], hist=data, init = weights_init)
   
    linear_regression.train(X_train, y_train, epochs=len(data["epochs"]), learning_rate=TrustOpinion.ftrust())
    
    X_arr = ArrayTO(TrustOpinion.fill(shape = (1, d), method="vacuous"))
    y_prime = linear_regression.forward_pass(X_arr)
    
    print("X")
    print(X_arr[0][0])
    print("Y")
     
    p = y_prime[0][0].t+y_prime[0][0].u*0.5
    print(y_prime[0][0],round(p, 3))
    
    X_arr = ArrayTO(TrustOpinion.fill(shape = (1, d), method="one", value=TrustOpinion(0,1,0)))
    y_prime = linear_regression.forward_pass(X_arr)
    
    print("X")
    print(X_arr[0][0])
    print("Y") 
    p = y_prime[0][0].t+y_prime[0][0].u*0.5
    print(y_prime[0][0],round(p, 3))

    X_arr = ArrayTO(TrustOpinion.fill(shape = (1, d), method="one"))
    y_prime = linear_regression.forward_pass(X_arr)
    print("X")
    print(X_arr[0][0])
    print("Y") 
    p = y_prime[0][0].t+y_prime[0][0].u*0.5
    print(y_prime[0][0],round(p, 3))


def check_par():
    # log_file = "californiaHouse/logs/a7cdf1aa-48d9-4b68-80f7-b3f72f24ef75.json"
    log_file = "californiaHouse/logs/a869f8f0-1f98-41ab-8adc-a8eb401476c9.json"

    data = parse_training_log_file(log_file)
    n_data = 20000
    d = 2
    X = TrustOpinion.fill(shape = (n_data, d), method="one", value=TrustOpinion(0,1,0))
    # X = TrustOpinion.fill(shape = (n_data, d))
    # print(X[0][0])
    y = TrustOpinion.fill(shape = (n_data, 1), method="one", value=TrustOpinion(1,0,0))
    # print (y[:5])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = ArrayTO(X_train), ArrayTO(X_test), ArrayTO(y_train), ArrayTO(y_test)

    # Create and train the linear regression model
    linear_regression = LinearRegression(input_dim=X_train.value.shape[1], hist=data, epsilon=1)
    linear_regression.train(X_train, y_train, epochs=len(data["epochs"]), learning_rate=TrustOpinion.ftrust())


    # one = TrustOpinion.fill(shape = (X_arr.value.shape[0], 1)) 
    # X_with_bias = ArrayTO(np.c_[X_arr.value, one])

    X_arr = ArrayTO(TrustOpinion.fill(shape = (1, d), method="vacuous"))
    y_prime = linear_regression.forward_pass(X_arr)
    
    print("X")
    print(X_arr)
    print("Y") 
    print(y_prime[0][0])


    X_arr = ArrayTO(TrustOpinion.fill(shape = (1, d), method="one", value=TrustOpinion(0,1,0)))
    y_prime = linear_regression.forward_pass(X_arr)
    
    print("X")
    print(X_arr)
    print("Y") 
    print(y_prime[0][0])


    X_arr = ArrayTO(TrustOpinion.fill(shape = (1, d), method="one"))
    y_prime = linear_regression.forward_pass(X_arr)
    
    print("X")
    print(X_arr)
    print("Y") 
    print(y_prime[0][0])


 

    
    # print(linear_regression.weights)
def check_sp():
    TrustOpinion.test_deduction()
    # log_file = "good/logs/a7cdf1aa-48d9-4b68-80f7-b3f72f24ef75.json"
    # a = parse_training_log_file(log_file)
    # print(len(a["epochs"]))
    # print(len(a["epochs"][0]["y"]))
    # print(len(a["epochs"][0]["y_prime"]))
    # print(len(a["epochs"][0]["loss"]))
# check_sp()
if __name__=='__main__':
    # check_par()  
    log_file = "californiaHouse/logs/a869f8f0-1f98-41ab-8adc-a8eb401476c9.json"
    eval(log_file)
    