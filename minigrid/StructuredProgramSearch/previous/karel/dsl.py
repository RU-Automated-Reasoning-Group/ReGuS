class k_prog:
    '''prog : DEF RUN M_LBRACE stmt M_RBRACE'''
    def __init__(self, stmt):
        self.stmt = stmt

    # find the leftmost place holder
    # return its handler
    def register(self):
        return self.stmt.register()

    def __call__(self, k, states_logger=None, exe_counter=None):
        # initial states
        # TODO: record meta info
        if states_logger:
            states_logger.init(k.state)
            states_logger.log(k.state)
        self.stmt(k, states_logger, exe_counter)

    def __str__(self):
        return "DEF RUN m( " + str(self.stmt) + " m)"


# place holder
class k_place_holder:
    '''place_holder : while
                    | stmt_stmt
                    | action
                    | if
                    | ifelse
    '''
    def __init__(self, index=None):
        self.index = index

    def __call__(self, k, states_logger=None, exe_counter=None):
        pass

    def __str__(self):
        return str("PH")


# non-expandable NULL
class k_null:
    '''non-expandable place holder, used to fix 
       problematic structure
    '''
    def __init__(self):
        pass

    def register(self):
        pass

    def __call__(self, k, states_logger=None, exe_counter=None):
        pass

    def __str__(self):
        return str('NULL')


class k_stmt:
    '''stmt : while
            | stmt_stmt
            | action
            | if
            | ifelse
    '''
    def __init__(self, function):
        self.function = function

    def register(self):
        if isinstance(self.function, k_place_holder):
            return self
        else:
            return self.function.register()

    def __call__(self, k, states_logger=None, exe_counter=None):
        if exe_counter:
            if not exe_counter.terminated():
                self.function(k, states_logger, exe_counter)
        else:
            self.function(k, states_logger, exe_counter)

    def __str__(self):
        return str(self.function)


class k_stmt_stmt:
    '''stmt_stmt : stmt stmt
    '''
    def __init__(self, stmt1, stmt2):
        self.stmt1 = stmt1
        self.stmt2 = stmt2

    def register(self):
        result1 = self.stmt1.register()
        result2 = self.stmt2.register()
        return result1 if result1 else result2

    def __call__(self, k, states_logger=None, exe_counter=None):
        if exe_counter:
            if not exe_counter.terminated():
                self.stmt1(k, states_logger, exe_counter)
                self.stmt2(k, states_logger, exe_counter)
        else:
            self.stmt1(k, states_logger, exe_counter)
            self.stmt2(k, states_logger, exe_counter)
    
    def __str__(self):
        return str(self.stmt1) + ' ' + str(self.stmt2)


class k_if:
    '''if : IF C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE
    '''
    def __init__(self, cond, stmt):
        self.cond = cond
        self.stmt = stmt

    def register(self):
        return self.stmt.register()

    def __call__(self, k, states_logger=None, exe_counter=None):
        if exe_counter:
            if not exe_counter.terminated():
                if self.cond(k):
                    self.stmt(k, states_logger, exe_counter)
        else:
            if self.cond(k):
                self.stmt(k, states_logger, exe_counter)
    
    def __str__(self):
        return "IF c( " + str(self.cond) + " c) i( " + str(self.stmt) + " i)"
    

class k_ifelse:
    '''ifelse : IFELSE C_LBRACE cond C_RBRACE I_LBRACE stmt I_RBRACE ELSE E_LBRACE stmt E_RBRACE
    '''
    def __init__(self, cond, stmt1, stmt2):
        self.cond = cond
        self.stmt1 = stmt1
        self.stmt2 = stmt2

    def register(self):
        result1 = self.stmt1.register()
        result2 = self.stmt2.register()
        return result1 if result1 else result2

    def __call__(self, k, states_logger=None, exe_counter=None):
        if exe_counter:
            if not exe_counter.terminated():
                if self.cond(k):
                    self.stmt1(k, states_logger, exe_counter)
                else:
                    self.stmt2(k, states_logger, exe_counter)
        else:    
            if self.cond(k):
                self.stmt1(k, states_logger, exe_counter)
            else:
                self.stmt2(k, states_logger, exe_counter)

    def __str__(self):
        return "IFELSE c( " + str(self.cond) + " c) i( " + str(self.stmt1) + " i) ELSE e( " + str(self.stmt2) + " e)"


class k_while:
    '''while : WHILE C_LBRACE cond C_RBRACE W_LBRACE stmt W_RBRACE
    '''
    def __init__(self, cond, stmt):
        self.cond = cond
        self.stmt = stmt

    def register(self):
        return self.stmt.register()

    def __call__(self, k, states_logger=None, exe_counter=None):
        if exe_counter:
            while(self.cond(k)):
                exe_counter.count_down()
                if not exe_counter.terminated():
                    self.stmt(k, states_logger, exe_counter)
                else:
                    break
        else:
            while(self.cond(k)):
                   self.stmt(k, states_logger, exe_counter)    

    def __str__(self):
        return "WHILE c( " + str(self.cond) + " c) w( " + str(self.stmt) + " w)"


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

    def register(self):
        return

    def __call__(self, k, states_logger=None, exe_counter=None):
        if exe_counter:
            exe_counter.count_down()
            if not exe_counter.terminated():
                getattr(k, self.action)()
                if states_logger:
                    states_logger.log(k.state)
                    # early termination
                    if states_logger.check(k.state):
                        exe_counter.zero()
        else:
            getattr(k, self.action)()
            if states_logger:
                states_logger.log(k.state)
        
    def __str__(self):
        return str(self.action)
