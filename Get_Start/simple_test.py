from ant_test import do_ant_test
from karel_test import test_large_maze
from highway_test import do_highway_test

do_ant_test()
test_large_maze()
try:
    do_highway_test()
    print('Simple Test Highway Case: Success')
except:
    print('Test Fail')