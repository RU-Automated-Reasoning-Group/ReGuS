# Tree structure to store search trace
import copy
import re

# from platform import node
import numpy as np
from minigrid_base_dsl import *

from utils.logging import log_and_print


# node in search tree
class SearchNode:
    def __init__(self, parent_node, current_sketch, cost, isleaf=False, score_C=1., cost_ratio=.5, lib_actions=None):
        # store
        self.parent = parent_node
        self.sketch = current_sketch
        self.isleaf = isleaf
        # init
        self.visit_num = 0
        self.visited = False
        self.children = None
        self.child_num = 0
        self.score_C = score_C
        self.cost_ratio = cost_ratio
        self.cost = cost
        self.score_list = []
        self.lib_actions = copy.deepcopy(lib_actions)

    # calculate structure cost
    # Dont use
    def get_cost(self):
        sketch_str = str(self.sketch)
        sketch_split = re.split('{|}|;| ', sketch_str)
        cost = sketch_split.count('S') + sketch_split.count('C') + sketch_split.count('SN')

        return cost

    # check if sketch contains B
    def check_B(self):
        sketch_str = str(self.sketch)
        sketch_split = re.split('{|}|;| |(|)', sketch_str)
        if 'B' in sketch_split:
            return True
        return False


    def get_score(self):
        if len(self.score_list) == 0:
            return -np.inf
        if self.parent is None:
            return np.max(self.score_list) - self.cost_ratio * self.cost
        return np.max(self.score_list) - self.cost_ratio * self.cost + self.score_C * np.sqrt(np.log(self.parent.visit_num) / self.visit_num)

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
                tmp_children.append(SearchNode(self, sketch, self.cost+1, isleaf))

            if self.parent is None:
                # root node (include library function as sketches)
                for lib_action in self.lib_actions:
                    # what should be the cost for this sketch (I don't think it's relevant because they cannot be expanded further)
                    tmp_children.append(SearchNode(self, lib_action, 1, False))
            
            # add
            if add_children:
                self.add_child(tmp_children)
            return tmp_children
        else:
            return None

    def __str__(self):
        return str(self.sketch)


# MCTS
class SearchTree:
    def __init__(self, root, alg, rollout_limit=10):
        self.root = root
        self.alg = alg
        self.rollout_limit = rollout_limit
        self.visited_sketch = {}

    # pick children of node with highest score or randomly
    def pick_children(self, node):
        # init
        best_child = None
        best_score = -np.inf
        nonvisited_children = []
        nonvisited_value = []
        
        # check all
        for child in node.children:
            if child.visited:
                if child.get_score() >= best_score:
                    best_child = child
                    best_score = child.get_score()
            else:
                nonvisited_children.append(child)
                nonvisited_value.append(child.cost)

        # pick
        if len(nonvisited_children) == 0:
            return best_child, best_score
        else:
            # return nonvisited_children[0], 0
            pick_id = np.argmin(nonvisited_value)
            # return np.random.choice(nonvisited_children), 0
            return nonvisited_children[pick_id], 0

    # traverse search tree to get new node (or leaf) for rollout
    def traverse(self, node=None):
        if node is None:
            node = self.root
        force_true = False
        # not leaf
        while (not node.isleaf and node.visited) or force_true:
            # expand node
            if node.child_num == 0:
                node.self_extend()
            # find next
            node, _ = self.pick_children(node)

            # check B
            if node.check_B():
                node.visited = True
                force_true = True
            else:
                force_true = False

        return node

    # policy (random) to rollout
    # No Use for Now
    def rollout_policy(self, node):
        # try extend
        children = node.self_extend(add_children=False)
        # for child in children:
        #     print(child)
        # pdb.set_trace()
        # randomly select
        return np.random.choice(children)

    # rollout from a node to reach leaf and get result
    def rollout(self, node):
        log_and_print('Current Sketch: {}'.format(node))
        # check
        # pdb.set_trace()
        approx_sketch = node.get_approx()
        if str(approx_sketch) in self.visited_sketch:
            log_and_print('find visited and apply previous result')
            assert node.visited is True
            reward, result_sketches = self.visited_sketch[str(approx_sketch)]
            pass_step = 0
        else:
            reward, result_sketches = self.alg.get_result(approx_sketch)
            self.visited_sketch[str(approx_sketch)] = [reward, result_sketches]
            node.visited = True
            pass_step = 1
        
        return node, reward, result_sketches, pass_step

    # backpropagation
    def backprop(self, node, reward):
        # init
        cur_node = node
        cur_node.score_list.append(reward)
        cur_node.visit_num += 1
        
        # add visit time for all parents
        while cur_node.parent is not None:
            cur_node = cur_node.parent
            cur_node.visit_num += 1
            cur_node.score_list.append(reward)

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
            best_score = np.max(cur_node.score_list) if len(cur_node.score_list) > 0 else -np.inf
            node_dict[cur_level].append((str(cur_node), cur_node.cost, cur_node.get_score(), cur_node.visit_num, best_score))
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


# TODO: modify traverse
# TODO: modify rollout