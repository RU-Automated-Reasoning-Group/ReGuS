    # WHILE(not (markers_present)) { 
    #   IF(not (front_is_clear)) { 
    #     turn_right{'front_is_clear': 'F', 'left_is_clear': 'DNC', 'right_is_clear': 'F', 'markers_present': 'F'} 
    #     turn_right{'front_is_clear': 'F', 'left_is_clear': 'F', 'right_is_clear': 'T', 'markers_present': 'F'}
    #   }  
    #   IF(right_is_clear) { 
    #     turn_right{'front_is_clear': 'T', 'left_is_clear': 'F', 'right_is_clear': 'T', 'markers_present': 'F'}
    #   }  
    #   move{'front_is_clear': 'T', 'left_is_clear': 'DNC', 'right_is_clear': 'DNC', 'markers_present': 'F'}
    # } ; S ; END

    # WHILE(not (markers_present)) { 
    #   IF(right_is_clear) { 
    #     turn_right{'front_is_clear': 'T', 'left_is_clear': 'F', 'right_is_clear': 'T', 'markers_present': 'F'}
    #   } 
    #   IF(not (front_is_clear)) { 
    #     turn_right{'front_is_clear': 'F', 'left_is_clear': 'DNC', 'right_is_clear': 'F', 'markers_present': 'F'} 
    #     turn_right{'front_is_clear': 'F', 'left_is_clear': 'F', 'right_is_clear': 'T', 'markers_present': 'F'}
    #   }   
    #   move{'front_is_clear': 'T', 'left_is_clear': 'DNC', 'right_is_clear': 'DNC', 'markers_present': 'F'}
    # } ; S ; END


    print(example_program)

    R_1 = ACTION(k_action('turn_right'))
    R_1_S = ABS_STATE()
    R_1_S.update('front_is_clear', 'F')
    R_1_S.update('left_is_clear', 'DNC')
    R_1_S.update('right_is_clear', 'F')
    R_1_S.update('markers_present', 'F')
    R_1.abs_state = R_1_S
    R_1.name = 'R_1'

    R_2 = ACTION(k_action('turn_right'))
    R_2_S = ABS_STATE()
    R_2_S.update('front_is_clear', 'F')
    R_2_S.update('left_is_clear', 'F')
    R_2_S.update('right_is_clear', 'T')
    R_2_S.update('markers_present', 'F')
    R_2.abs_state = R_2_S
    R_2.name = 'R_2'

    R_3 = ACTION(k_action('turn_right'))
    R_3_S = ABS_STATE()
    R_3_S.update('front_is_clear','T')
    R_3_S.update('left_is_clear', 'F')
    R_3_S.update('right_is_clear', 'T')
    R_3_S.update('markers_present', 'F')
    R_3.abs_state = R_3_S
    R_3.name = 'R_3'

    M_1 = ACTION(k_action('move'))
    M_1_S = ABS_STATE()
    M_1_S.update('front_is_clear','T')
    M_1_S.update('left_is_clear', 'DNC')
    M_1_S.update('right_is_clear', 'DNC')
    M_1_S.update('markers_present', 'F')
    M_1.abs_state = M_1_S
    M_1.name = 'M_1'

    IF_1 = IF(cond=COND_DICT['not(front_is_clear)'])
    IF_1.stmts = [R_1, R_2] 

    IF_2 = IF(cond=COND_DICT['right_is_clear'])
    IF_2.stmts = [R_3]

    example_program.stmts[0].stmts = [
        IF_2,
        IF_1,
        M_1
    ]
    
    print(example_program)
    robot = KarelRobot(task='randomMaze', seed=0)
    robot.force_execution = True
    robot.draw()
    example_program.execute(robot)
    robot.draw()
    print(robot.steps)
    exit()


########
########
#...#.##
###1#.##
#.#...##
#.#.####
#>....##
########