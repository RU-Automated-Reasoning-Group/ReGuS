# class k_prog:
#     '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
#     def __init__(self):
#         self.stmts = [k_place_holder()]

#     def __call__(self, k, robot=None):
#         for s in self.stmts:
#             s(k, robot)

#     def __str__(self):
#         stmts_str = ''
#         for s in self.stmts:
#             stmts_str += str(s)
#             stmts_str += ' '
#         return "DEF RUN m( " + stmts_str + " m)"
#         #return "DEF RUN m( " + str(self.stmts) + " m)"


# # place holder
# class k_place_holder:
#     '''place_holder : while
#                     | stmt_stmt
#                     | action
#                     | if
#                     | ifelse
#     '''
#     def __init__(self, candidates=None, end_for_while=False, cond=None):
#         self.candidates = candidates
        
#         # used only in while loop
#         self.end_for_while = end_for_while
#         self.cond = cond

#     def __call__(self, k, robot=None):
#         pass

#     def __str__(self):
#         return str("PH")


# class k_end:
#     '''end of current scope
#     '''
#     def __init__(self, dummy=False):
#         self.dummy = dummy

#     def __call__(self, k, robot=None):
        
#         # TODO: a compromised solution for useless structure
#         # TODO: can be improved, 
#         #          if() {k_end()}
#         #          ifelse() {...} else {k_end()}
#         if robot:
#             robot.steps += 1
#             if robot.steps >= robot.max_steps:
#                 robot.no_fuel = True
        
#         if not robot.no_fuel:
#             if self.dummy:
#                 return robot.check_reward()
#         else:
#             return 0

#     def __str__(self):
#         return 'end'


# class k_if:
#     '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
#     '''
#     def __init__(self):
#         self.cond = None
#         self.stmts = []

#     def __call__(self, k, robot=None):
#         if not robot.no_fuel:
#             if self.cond(k):
#                 for s in self.stmts:
#                     s(k, robot)
    
#     def requires_cond(self):
#         return not isinstance(self.cond, k_cond)

#     def finished(self):
#         return self.requires_cond and len(self.stmts) >=1 and isinstance(self.stmts[-1], k_end)


#     def __str__(self):
#         return "IF c( " + str(self.cond) + " c) i( " + str(self.stmts) + " i)"
    

# class k_ifelse:
#     '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
#     '''
#     def __init__(self):
#         self.cond = None
#         self.stmts1 = []
#         self.stmts2 = []

#     def __call__(self, k, robot=None):
#         if not robot.no_fuel:
#             if self.cond(k):
#                 for s in self.stmts1:
#                     s(k, robot)
#             else:
#                 for s in self.stmts2:
#                     s(k, robot)

#     def requires_cond(self):
#         return not isinstance(self.cond, k_cond)

#     def finished(self):
#         return self.requires_cond and \
#             len(self.stmts1) >=1 and isinstance(self.stmts1[-1], k_end) and \
#             len(self.stmts2) >=1 and isinstance(self.stmts2[-1], k_end) \

#     def __str__(self):
#         return "IFELSE c( " + str(self.cond) + " c) i( " + str(self.stmts1) + " i) ELSE e( " + str(self.stmts2) + " e)"


# class k_while:
#     '''while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE
#     '''
#     def __init__(self):
#         self.cond = None
#         self.stmts = []
        
#     def __call__(self, k, robot=None):
#         if not robot.no_fuel:    
#             if not self.useless():
#                 while self.cond(k) and not robot.no_fuel:
#                     assert not self.useless()
#                     for s in self.stmts:
#                         s(k, robot)
        
#     def requires_cond(self):
#         return not isinstance(self.cond, k_cond)

#     def finished(self):
#         return self.requires_cond and len(self.stmts) >=1 and isinstance(self.stmts[-1], k_end)

#     def useless(self):
#         return len(self.stmts) == 1 and isinstance(self.stmts[-1], k_end)
 
#     def __str__(self):
#         stmts_str = ''
#         for s in self.stmts:
#             stmts_str += str(s)
#             stmts_str += ' '
#         return "WHILE c( " + str(self.cond) + " c) w( " + stmts_str + " w)"
#         #return "WHILE c( " + str(self.cond) + " c) w( " + str(self.stmts) + " w)"


class k_cond:
    '''cond : cond_without_not
            | NOT C_LBRACE cond_without_not C_RBRACE
    '''
    def __init__(self, negation: bool, cond):
        self.negation = negation
        self.cond = cond

    def __call__(self, k):
        if self.negation:
            return not self.cond(k)
        else:
            return self.cond(k)

    def __str__(self):
        if self.negation:
            return "NOT c( " + str(self.cond) + " c)"
        else:
            return str(self.cond)


class k_cond_without_not:
    '''cond_without_not : FRONT_IS_CLEAR
                        | LEFT_IS_CLEAR
                        | RIGHT_IS_CLEAR
                        | MARKERS_PRESENT
                        | NO_MARKERS_PRESENT
    '''
    def __init__(self, cond: str):
        self.cond = cond

    def __call__(self, k):
        return getattr(k, self.cond)()

    def __str__(self):
        return str(self.cond)


class k_action:
    '''action : MOVE
              | TURN_RIGHT
              | TURN_LEFT
              | PICK_MARKER
              | PUT_MARKER
    '''
    def __init__(self, action: str):
        self.action = action

    def __call__(self, k, robot=None):

        if not robot.no_fuel():
            getattr(k, self.action)()
            if robot:
                robot.steps += 1
                return robot.check_reward()
                #if robot.steps >= robot.max_steps:
                #    robot.no_fuel = True
                # reward = robot.check_reward()
                # if robot.is_accumulated:
                #     robot.acc(reward)
                # return reward
        else:
            #print('?')
            #return 0
            return robot.check_reward()
        #else:
        #    # TODO: if not fuel during restart
        #    return 0
        
    def __str__(self):
        return str(self.action)

