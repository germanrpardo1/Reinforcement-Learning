import numpy as np
import matplotlib.pyplot as plt
import copy
from operator import itemgetter
from collections import deque
import random
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()
class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height

        # Define the tensorflow neural network
        # 1. Input:
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 1, board_width, board_height])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        # 2. Common Networks Layers
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            data_format="channels_last",
                                            activation=tf.nn.relu)
        # Flatten the tensor
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run([self.action_fc, self.evaluation_fc2], feed_dict={self.input_states: state_batch})
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, env):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        current_state = np.ascontiguousarray(env.board.reshape(-1, 1, self.board_width, self.board_height))

        act_probs, value = self.policy_value(current_state)
        act_probs = act_probs[0]
        
        probs = np.zeros(len(env.initial_actions))
        r_probs = probs
        for i in range(len(env.initial_actions)):
            if env.initial_actions[i] in env.actions:
                probs[i] = act_probs[i]
        if np.sum(probs) != 0:
            probs /= np.sum(probs)
            r_probs = zip(env.initial_actions, probs)
        return r_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)



class Board_AlphaGo:
    def __init__(self, rows, cols):
        self.rows = rows
        self.board = np.zeros((self.rows, self.rows))
        self.reward = 0.0
        self.isTerminal = False
        self.initial_actions = self.availablePositions()
        self.actions = self.initial_actions
        self.player = 1

        # Buffer to save self-play data
        self.buffer_size = 10000
        self.batch_size = 128  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)


        self.learn_rate = 0.01
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.kl_targ = 0.02
        
    def winner(self):
        # row
        self.isTerminal = True
        for i in range(self.rows):
            if sum(self.board[i, :]) == self.rows:
                return 1
            if sum(self.board[i, :]) == -self.rows:
                return -1
        # col
        for i in range(self.rows):
            if sum(self.board[:, i]) == self.rows:
                return 1
            if sum(self.board[:, i]) == -self.rows:
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(self.rows)])
        diag_sum2 = sum([self.board[i, self.rows - i - 1] for i in range(self.rows)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == self.rows:
            if diag_sum1 == self.rows or diag_sum2 == self.rows:
                return 1
            else:
                return -1
        # tie
        # no available positions
        if len(self.actions) == 0:
            return 0
        # not end
        self.isTerminal = False
        return 0

    def availablePositions(self):
        positions = []
        for i in range(self.rows):
            for j in range(self.rows):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def step(self, action):
        self.board[action] = self.player
        self.actions = self.availablePositions()
        self.reward = self.winner()
        if self.actions == []:
            self.isTerminal = True
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1                
        return self.board, self.actions, self.reward, self.isTerminal

    # board reset
    def reset(self):
        self.player = 1
        self.board = np.zeros((self.rows, self.rows))
        self.actions = self.initial_actions
        self.reward = 0.0
        self.isTerminal = False
        return self.board, self.actions, self.isTerminal

    def showBoard(self):
        for i in range(0, self.rows):
            print('----'*self.rows)
            out = '| '
            for j in range(0, self.rows):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('----'*self.rows)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        
        # To recover the original tree structure with all the information
        self._exparent = None
        
    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                if prob > 0.0:
                    self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value) #-leaf_value?
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return (self._Q + self._u)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            env.step(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(env)
        # Check for end of game.
        if not env.isTerminal:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            leaf_value = env.reward

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)#-leaf_value?

    def get_move_probs(self, env, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            env_copy = copy.deepcopy(env)
            self._playout(env_copy)
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs


    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        node = self._root
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._exparent = self._root._parent
            self._root._parent = None
        else:
            while node._exparent != None:
                prev_node = node
                node = node._exparent
                prev_node._parent = node 
            self._root = node
            #self._root = TreeNode(None, 1.0)

class MCTSPlayer(object):
    """AI player based on MCTS following the AlphaZero approach"""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=1e-3, return_prob=False):
        sensible_moves = env.actions
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(env.rows ** 2)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)
            #move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                acts_index = [i for i in range(len(acts))]
                index_act = np.random.choice(acts_index, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                move = acts[index_act]
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                acts_index = [i for i in range(len(acts))]
                index_act = np.random.choice(acts_index, p = probs)
                move = acts[index_act]
                # reset the root node
                self.mcts.update_with_move(move) #-1?

            if return_prob:
                return move, move_probs
            else:
                return move