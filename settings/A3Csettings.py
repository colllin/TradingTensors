#Choose either Shared or Separate AC Network
SHARED = False

#Separate Network
ACTOR_LAYER = [64]
CRITIC_LAYER = [128]

#Shared Network
SHARED_LAYER = [128, 64]

#Hyperparams
<<<<<<< Updated upstream
NUM_WORKERS = 8
GAMMA = 0.99
=======
NUM_WORKERS = 4
GAMMA = 0.9
>>>>>>> Stashed changes
LEARNING_RATE = 0.001

#Regularization terms to control overfitting
COEF_REG = 0.5
L2_REG_LAMBDA = 0.1

UPDATE_FREQ = 50
