from robot_dsl import *

class ConvertProg:
    def __init__(self, dsl_dict):
        self.dsl_dict = dsl_dict

    # get list of actions
    def _get_prog_acts(self, prog):
        # init
        prog_list = prog.split(' ')
        act_list = []

        for act_str in prog_list:
            if len(act_str) == 0 or act_str == ';;':
                continue
            # if branch
            if 'IF' in act_str: 
                cond = act_str[3:-1]
                new_act= self.dsl_dict['IF'](self.dsl_dict[cond])
                act_list.append(new_act)

            # while branch
            elif 'WHILE' in act_str:
                cond = act_str[6:-1]
                new_act= self.dsl_dict['WHILE']()
                new_act.cond = [self.dsl_dict[cond]]
                act_list.append(new_act)

            # else branch
            elif 'ELSE' in act_str:
                act_list.append('ELSE')

            # action
            else:
                act_list.append(self.dsl_dict[act_str])

        return act_list

    # find paired '{}' and return
    def _get_prog_help(self, prog):
        # init
        left_num = 0
        statement_list = []

        # get statement
        while left_num >= 0:
            # init
            prog = prog.strip()

            # get left
            left_idx = None
            if '{' in prog:
                left_idx = prog.index('{')
            # get right
            right_idx = len(prog)
            if '}' in prog:
                right_idx = prog.index('}')
            # update
            if left_idx is not None and left_idx < right_idx:
                left_num += 1

            # all action list
            if left_idx is None or left_idx > right_idx:
                act_list = self._get_prog_acts(prog[:right_idx])
                prog = prog[right_idx+1:]
                # store and next
                statement_list += act_list
                left_num -= 1
            # need new statement
            else:
                act_list = self._get_prog_acts(prog[:left_idx])
                # prog = prog[left_idx+1:]
                new_statement, prog = self._get_prog_help(prog[left_idx+1:])
                # store and next
                if act_list[-1] == 'ELSE':
                    last_statement = statement_list[-1]
                    assert isinstance(last_statement, self.dsl_dict['IF'])
                    replace_statement = self.dsl_dict['IFELSE'](last_statement.cond[0])
                    replace_statement.stmts = last_statement.stmts
                    replace_statement.else_stmts = new_statement
                    # store and next
                    statement_list[-1] = replace_statement
                    left_num -= 1
                else:
                    act_list[-1].stmts = new_statement
                    # store and next
                    statement_list += act_list
                    left_num -= 1

        return statement_list, prog


    # TODO: test
    def get_prog(self, prog):
        prog, prog_left = self._get_prog_help(prog)
        out_prog = Program()
        out_prog.stmts = prog

        return out_prog