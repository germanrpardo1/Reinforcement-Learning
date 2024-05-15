import numpy as np

class BoardQ:
    def __init__(self, rows, cols):
        self.rows = rows
        self.board = np.zeros((rows, cols))
        self.state = str(self.board.reshape(rows * cols))
        self.reward = 0.0
        self.isTerminal = False
        self.initial_actions = self.availablePositions()
        self.actions = self.initial_actions
        self.state_actions = {}
        
    def winner(self):
        # row
        for i in range(self.rows):
            if sum(self.board[i, :]) == self.rows:
                self.isTerminal = True
                return 1
            if sum(self.board[i, :]) == -self.rows:
                self.isTerminal = True
                return -1
        # col
        for i in range(self.rows):
            if sum(self.board[:, i]) == self.rows:
                self.isTerminal = True
                return 1
            if sum(self.board[:, i]) == -self.rows:
                self.isTerminal = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(self.rows)])
        diag_sum2 = sum([self.board[i, self.rows - i - 1] for i in range(self.rows)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == self.rows:
            self.isTerminal = True
            if diag_sum1 == self.rows or diag_sum2 == self.rows:
                return 1
            else:
                return -1
        # tie
        # no available positions
        if len(self.actions) == 0:
            self.isTerminal = True
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
        self.board[action] = 1
        self.actions = self.availablePositions()
        self.state = str(self.board.reshape(self.rows ** 2))
        self.reward = self.winner()
        if not self.isTerminal and self.actions != []:
            self.random_policy()
                
        return self.state,  self.actions, self.reward, self.isTerminal

    def random_policy(self):
        # Random opponent
        opp_action = np.random.randint(0, len(self.actions))
        self.board[self.actions[opp_action]] = -1
        self.actions = self.availablePositions()
        self.state = str(self.board.reshape(self.rows ** 2))
        self.reward = self.winner()

    # board reset
    def reset(self):
        self.board = np.zeros((self.rows, self.rows))
        self.state = str(self.board.reshape(self.rows ** 2))
        self.actions = self.initial_actions
        self.reward = 0.0
        self.isTerminal = False
        
        return self.state, self.actions, self.isTerminal
        
    def add_avaliable_actions(self, s, q):
        self.state_actions[s] = self.actions
        for new_a in self.state_actions[s]:
            q[s, new_a] = 0
        return q

    def showBoard(self):
        for i in range(0, self.rows):
            print('----'* self.rows)
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
        print('----'*rows)