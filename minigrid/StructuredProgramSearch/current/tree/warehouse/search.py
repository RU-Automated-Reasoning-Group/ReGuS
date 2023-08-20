if _robot.execute_single_cond(c_cond) and cond_type == 'w':
                
                c_stmts, c_idx = candidate.find_actions()
                tmp_action.abs_state = tmp_abs_state
                tmp_action.post_abs_state = tmp_post_abs_state
                c_stmts[c_idx-1] = tmp_action
                c_stmts.pop(c_idx)  # remove C

                # TODO: fix this
                #candidate.execute(tmp_robot, start_at=None)

                eval_robot = copy.deepcopy(original_robot)
                candidate.execute(eval_robot)  # TODO: stop when reward is 1 or -1?
                r = eval_robot.check_reward()
                if r == 1:
                    print('[we found it, w]', candidate)
                    #exit()
                    node.candidates.append(('success', 1, candidate))
                elif eval_robot.no_fuel():
                    print('[no fuel, w]')
                    node.candidates.append(('no_fuel', r, candidate))
                elif candidate.complete():
                    print('[complete, w]')
                    node.candidates.append(('complete', r, candidate))
                else:
                    # reactivate robot & reset program
                    eval_robot.active = True
                    candidate.reset()

                    # find break point
                    bp_stmts, bp_idx = candidate.find_break_point()
                    bp = bp_stmts[bp_idx]

                    # determine cond at break_point IF
                    diff_conds = diff_abs_state(bp.abs_state, bp.diff_abs_state)
                    
                    # insert IF(cond) {C} at break point
                    for j in range(len(diff_conds)):
                        if j == 0:
                            bp_stmts.insert(bp_idx, IF(cond=diff_conds[j]))
                            print('[get a new program, first, w]', candidate)
                        else:
                            bp_stmts[bp_idx] = IF(cond=diff_conds[j])
                            print('[get a new program, more than one, w]', candidate)

                        #print('[before put][1]', candidate)
                        
                        q_tuple = (
                            -r,
                            tmp_cost,
                            time.time(),
                            copy.deepcopy(candidate),
                            {
                                #'program': candidate,
                                'robot': eval_robot,
                                'rules': tmp_rules,  # TODO: should we include IF in rules?
                            }
                        )
                        node.q.put(q_tuple)
                    
            elif cond_type == 'i':

                # find break point
                bp_stmts, bp_idx = candidate.find_break_point()
                bp = bp_stmts[bp_idx]
                
                c_stmts, c_idx = candidate.find_actions()
                tmp_action.abs_state = tmp_abs_state
                tmp_action.post_abs_state = tmp_post_abs_state
                c_stmts[c_idx-1] = tmp_action

                # try to connect to the break point
                future_robot = copy.deepcopy(_robot)  # for on time use
                current_abs_state = get_abs_state(future_robot)
                future_robot.execute_single_action(bp.action)  # execute break_point action
                future_abs_state = get_abs_state(future_robot)
                
                if satisfy_abs_state(future_abs_state, bp.post_abs_state):

                    bp.abs_state = merge_abs_state(bp.abs_state, current_abs_state)
                    bp.break_point = False
                    c_stmts.pop(c_idx)  # remove C

                    eval_robot = copy.deepcopy(original_robot)
                    candidate.execute(eval_robot)
                    r = eval_robot.check_reward()
                    if r == 1:
                        print('[we found it, i]', candidate)
                        #exit()
                        node.candidates.append(('success', 1, candidate))
                    elif eval_robot.no_fuel():
                        print('[no fuel, i]')
                        node.candidates.append(('no_fuel', r, candidate))
                    elif candidate.complete():
                        print('[complete, i]')
                        node.candidates.append(('complete', r, candidate))
                    else:

                        print('[restart and need new branch, i]')
                        
                        # reactivate robot & reset program
                        eval_robot.active = True
                        candidate.reset()

                        # find & deactivate break point
                        bp_stmts, bp_idx = candidate.find_break_point()
                        bp = bp_stmts[bp_idx]
                        bp.break_point = False

                        # determine cond at break_point IF
                        diff_conds = diff_abs_state(bp.abs_state, bp.diff_abs_state)
                        
                        # insert IF(cond) {C} at break point
                        for j in range(len(diff_conds)):
                            if j == 0:
                                bp_stmts.insert(bp_idx, IF(cond=diff_conds[j]))
                                print('[get a new program, first, i]', candidate)
                            else:
                                bp_stmts[bp_idx] = IF(cond=diff_conds[j])
                                print('[get a new program, more than one, i]', candidate)
                            
                            q_tuple = (
                                -r,
                                tmp_cost,
                                time.time(),
                                copy.deepcopy(candidate),
                                {
                                    'robot': eval_robot,
                                    'rules': tmp_rules,  # TODO: should we include IF in rules?
                                }
                            )
                            node.q.put(q_tuple)
                
                else:

                    print('[not really success, continue to expand, i]')

                    c_stmts, c_idx = candidate.find_actions()
                    c_stmts[c_idx-1].abs_state = tmp_abs_state  # NOTE: includes abs_state
                    c_stmts[c_idx-1].post_abs_state = tmp_post_abs_state
                    
                    q_tuple = (
                        -tmp_r,  # primary: retreive item with the highest reward
                        tmp_cost,  # secondary: retreive item with the lowest cost
                        time.time(),  # the final evaluation metric
                        copy.deepcopy(candidate),
                        {
                            'robot': _robot, #tmp_robot,
                            'rules': tmp_rules,
                        }
                    )
                    
                    node.q.put(q_tuple)

            else:

                print('[expand w further]')

                c_stmts, c_idx = candidate.find_actions()
                c_stmts[c_idx-1].abs_state = tmp_abs_state
                c_stmts[c_idx-1].post_abs_state = tmp_post_abs_state
                
                q_tuple = (
                    -tmp_r,  # primary: retreive item with the highest reward
                    tmp_cost,  # secondary: retreive item with the lowest cost
                    time.time(),  # the final evaluation metric
                    copy.deepcopy(candidate),
                    {
                        'robot': _robot,
                        'rules': tmp_rules,
                    }
                )
                
                node.q.put(q_tuple)





    # NOTE: this does not work
    # TODO: this should be fixed
    def before_find_c_cond(self, stmts):
        cond, cond_type = None, None
        for idx, code in enumerate(stmts):
            if isinstance(code, (S, B, ACTION, END)):
                pass
            elif isinstance(code, (WHILE, IF)):
                for w_idx, s in enumerate(code.stmts):
                    if isinstance(s, C):
                        return code.cond[0], 'w' if isinstance(code, WHILE) else 'i'
                    elif isinstance(s, ACTION):
                        pass
                    else:
                        cond, cond_type = self._find_c_cond(code.stmts)
                        if cond is not None:
                            return cond, cond_type
            else:
                raise ValueError('Invalid code')

        return cond, cond_type