class k_prog:
    '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
    def __init__(self):
        self.stmt = []

    def append(self, action):
        self.stmt.append(action)

    def __call__(self, k, robot=None):
        for action in self.stmt:
            action(k, robot=None)

    def __str__(self):
        return "DEF RUN m( " + str(self.stmt) + " m)"



class k_action:
    '''action : MOVE
              | TURN_RIGHT
              | TURN_LEFT
              | PICK_MARKER
              | PUT_MARKER
    '''
    def __init__(self, action: str):
        self.action = action

    def register_ph(self):
        return

    def register_while(self):
        return []

    def __call__(self, k, robot=None):
        getattr(k, self.action)()
        if robot:
            return robot.check_reward()
        
    def __str__(self):
        return str(self.action)
