class DNQAgent:
    def __init__(self, model):
        self.model = model # NN

    def get_actions(self, observations):
        # observation shape is (N, 4) N: batch size
        q_vals = self.model(observations) # this does forward propagation
        # q_vals shape is (N,2)
        TODO
        return q_vals.max(-1) # ?!?!?!?!?!?!!?