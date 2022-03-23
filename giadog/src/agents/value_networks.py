"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]

"""
from tensorflow import keras
from tensorflow.keras import layers

class TeacherValueNetwork:
    """
    Teacher value network class.
    """
    
    def __init__(self):
        """
        Initializes the teacher value network.
        """
    
        self.PRIVILIGED_DATA_SHAPE = 59
        self.NON_PRIVILIGED_DATA_SHAPE = 145
        
        inputs_x_t = keras.Input(
            shape=self.PRIVILIGED_DATA_SHAPE, 
            name='priviliged_data'
        )

        inputs_o_t = keras.Input(
            shape=self.NON_PRIVILIGED_DATA_SHAPE,
            name='non_priviliged_data')


        concat = layers.Concatenate()([inputs_x_t, inputs_o_t])
        
        x = layers.Dense(256, activation='tanh')(concat)
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Dense(16, activation='tanh')(x)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs =  [inputs_x_t, inputs_o_t], 
                                 outputs = outputs, 
                                 name='value network')
    
    def __call__(self, state):
        """
        Computes the value of the state.

        Arguments:
        ----------
        state: np.array -- The state of the environment.
        """
        
        return self.model(state)

    def save(self, path: str):
        """
            Saves the model weights to a directory

            Arguments:
            ----------
                path: str 
                    Path where the model will be saved.
        """
        self.model.save_weights(path)

    def load(self, path: str):
        """
            Loads the model weights from a directory.

            Arguments:
            ----------
                path: str 
                    Path to the file where the weights will be loaded.
        """
        self.model.load_weights(path)
