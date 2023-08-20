# Tree structure to store search trace
import copy
import heapq
import pdb
import re
import time

# from platform import node
import numpy as np
from dsl import *

from utils.logging import log_and_print


# node in search tree
class SearchNode:
    def __init__(self, parent_node, current_sketch, g_cost, isleaf=False, cost_ratio=0.5):
        # store
        self.parent = parent_node
        self.sketch = current_sketch
        self.isleaf = isleaf
        # init
        self.children = None
        self.child_num = 0
        self.g_cost = g_cost
        self.f_cost = None
        self.cost_ratio = cost_ratio
        # get structure cost as g_score
        # self.g_cost = self.get_cost()

    # get total cost
    def get_total_cost(self):
        if self.f_cost is not None:
            return self.cost_ratio * self.g_cost + self.f_cost
        else:
            return self.cost_ratio * self.g_cost

    # check if sketch contains B
    def check_B(self):
        sketch_str = str(self.sketch)
        sketch_split = re.split('{|}|;| |(|)', sketch_str)
        if 'B' in sketch_split:
            return True
        return False

    # calculate structure cost
    # DONT USE
    def get_cost(self):
        # init
        sketch_str = str(self.sketch)
        depth = 0
        cur_depth = 0
        # find max depth
        for c in sketch_str:
            if c == '{':
                cur_depth += 1
                if cur_depth > depth:
                    depth = cur_depth
            elif c == '}':
                assert cur_depth > 0
                cur_depth -= 1
        # get cost
        assert cur_depth == 0
        return depth


    # get approximate sketch
    def get_approx(self):
        approx_sketch = copy.deepcopy(self.sketch)
        stmt_list = [approx_sketch.stmts]
        # change items
        while len(stmt_list) > 0:
            cur_stmt = stmt_list.pop(0)
            for idx, code in enumerate(cur_stmt):
                if isinstance(code, (WHILE, IF)):
                    stmt_list.append(code.stmts)
                elif isinstance(code, (S, SN)):
                    cur_stmt[idx] = C()
        
        return approx_sketch


    def add_child(self, children_node):
        if self.children is None:
            self.children = []
        self.children += children_node
        self.child_num += len(children_node)

    def self_extend(self, add_children=True):
        if not self.isleaf:
            tmp_sketch = copy.deepcopy(self.sketch)
            tmp_children_sketch = tmp_sketch.expand()
            # pdb.set_trace()
            tmp_children = []

            # get node
            for sketch in tmp_children_sketch:
                # try extend
                if len(copy.deepcopy(sketch).expand()) == 0:
                    isleaf = True
                else:
                    isleaf = False
                # create node
                tmp_children.append(SearchNode(self, sketch, self.g_cost+1, isleaf))

            # add
            if add_children:
                self.add_child(tmp_children)

            return tmp_children
        else:
            return None

    def __str__(self):
        return str(self.sketch)


# A* search
class SearchTree:
    def __init__(self, root, alg):
        self.root = root
        self.alg = alg
        self.search_queue = [(None, None, self.root)]
        self.visited_sketch = {}

    # do search for one step
    def expand(self, node=None):
        # pick node with least cost
        if node is None:
            if len(self.search_queue) > 0:
                _, _, node = heapq.heappop(self.search_queue)
                assert node.child_num == 0
            else:
                log_and_print('queue is empty')
                return None
        
        # expand node
        if node.child_num == 0:
            node.self_extend()
        log_and_print('sketch based on: {}'.format(str(node)))

        # get child nodes to be searched
        search_node = []
        for child in node.children:
            if child.check_B():
                child.f_cost = node.f_cost
                heapq.heappush(self.search_queue, (child.get_total_cost(), time.time(), child))
            else:
                search_node.append(child)
        if len(search_node) == 0:
            return None, 1-node.f_cost, None, 0

        np.random.shuffle(search_node)

        # get reward
        best_node = None
        best_sketches = None
        best_reward = -100
        used_step = 0
        for child in search_node:
            used_step += 1
            # leaf node are all been visited before
            if child.isleaf:
                child.f_cost = self.visited_sketch[str(child)]
                continue

            approx_sketch = child.get_approx()
            # check whether visited
            if str(approx_sketch) in self.visited_sketch:
                child.f_cost = self.visited_sketch[str(approx_sketch)]
                heapq.heappush(self.search_queue, (child.get_total_cost(), time.time(), child))
                continue

            start_time = time.time()
            log_and_print('current sketch: {}\n{}\n'.format(str(child), str(approx_sketch)))
            reward, result_sketches = self.alg.get_result(child.get_approx())
            log_and_print('Time used for current step: {}'.format(time.time()-start_time))

            child.f_cost = 1-reward
            self.visited_sketch[str(approx_sketch)] = child.f_cost
            heapq.heappush(self.search_queue, (child.get_total_cost(), time.time(), child))

            # store
            if reward > best_reward:
                best_node = child
                best_sketches = result_sketches
                best_reward = reward
            if best_reward == 1:
                break

        return best_node, best_reward, best_sketches, used_step

    # print all nodes inside search tree
    def display_all(self):
        # init
        cur_node = self.root
        node_dict = {}
        queue = [(cur_node, 0)]
        # BFS to get all nodes
        while len(queue) != 0:
            # get
            cur_node, cur_level = queue.pop(0)
            if cur_level not in node_dict:
                node_dict[cur_level] = []

            node_dict[cur_level].append((str(cur_node), cur_node.g_cost, cur_node.f_cost, cur_node.get_total_cost()))
            # add into queue
            child_nodes = cur_node.children
            if child_nodes is not None:
                for next_node in child_nodes:
                    queue.append((next_node, cur_level+1))

        # print
        for level in node_dict:
            log_and_print('for level {}'.format(level))
            for node in node_dict[level]:
                log_and_print('{}'.format(node))
            log_and_print('----------------------------')