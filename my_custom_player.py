
from sample_players import DataPlayer
# from isolation import DebugState ## Uncomment if need to debug

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """   
    def alpha_beta_search(self, state, depth=3, heuristic_name="custom"):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        Ignores the special case of calling this function
        from a terminal state.
        """
        def index2xy(ind):
            """ Convert from board index value to xy coordinates
            ## Note: Taken from DebugState.ind2xy because submit didn't work when calling that
            The coordinate frame is 0 in the bottom right corner, with x increasing
            along the columns progressing towards the left, and y increasing along
            the rows progressing towards teh top.
            """
            _WIDTH = 11 # Not dynamic here because using DebugState.ind2xy function
            return (ind % (_WIDTH + 2), ind // (_WIDTH + 2))
        
        def distanceToCenter(location):
            """
            Find the distance of location to the center square which is (5, 4) or index 57
            """
            #from isolation import DebugState 
            x1, y1 = index2xy(location) #DebugState.ind2xy(location)
            x2, y2 = 5, 4   # DebugState.indxy(57) = (5, 4)
            distance = ( (x1-x2)**2 + (y1-y2)**2 )**(1/2)
            return distance

        def baseline_heuristic(gameState):
            """ A heuristic to be used as a baseline. Return the
            score for current state based on only the combination
            of own liberties and opponent liberties.
            """
            own_loc = gameState.locs[self.player_id]
            opp_loc = gameState.locs[1 - self.player_id]
            own_liberties = gameState.liberties(own_loc)
            opp_liberties = gameState.liberties(opp_loc)
            return len(own_liberties) - len(opp_liberties)

        def greedy_heuristic(gameState):
            """ A greedy heuristic. Return the score for current 
            state based on only the number of own player's liberties.
            """
            loc = gameState.locs[self.player_id]
            liberties = gameState.liberties(loc)
            return len(liberties)

        def custom_heuristic(gameState):
            """ A custom heuristic. Return the score for current 
            state based on both a weighted distance of the player 
            to the center square and a weighted ratio between the
            player's and opponent's liberties at the current move. 
            
            Comments list out other custom heuristics tested. 
            """
            center_weight = 0.5
            lib_weight = 1.5
            own_loc = gameState.locs[self.player_id]
            opp_loc = gameState.locs[1- self.player_id]
            own_liberties = gameState.liberties(own_loc)
            opp_liberties = gameState.liberties(opp_loc)
            # Custom 1: distanceToCenter(own_loc)
            # Custom 2: len(own_liberties) - ( center_weight * distanceToCenter(own_loc) )
            # Custom 3: len(own_liberties) - ( len(opp_liberties) ) - ( center_weight * distanceToCenter(own_loc) ) 
            # Custom 4: len(own_liberties) - ( lib_weight * len(opp_liberties) ) - ( center_weight * distanceToCenter(own_loc) )
            # Custom 5: ( lib_weight * (len(own_liberties) / len(opp_liberties)) - ( center_weight * distanceToCenter(own_loc)) )
            return ( lib_weight * (len(own_liberties) / len(opp_liberties)) - (center_weight * distanceToCenter(own_loc)) )

        def min_value(gameState, alpha, beta, depth, heuristic_name):
            """ Return the value for a win (+1) if the game is over,
            otherwise return the minimum value over all legal child
            nodes.
            """
            if gameState.terminal_test():
                return gameState.utility(self.player_id) 
            if depth <= 0:
                if heuristic_name == "custom":
                    return custom_heuristic(gameState)
                elif heuristic_name == "baseline":
                    return baseline_heuristic(gameState)
                elif heuristic_name == "greedy":
                    return greedy_heuristic(gameState)
                else:
                    return custom_heuristic(gameState) 
    
            v = float("inf")
            for a in gameState.actions():
                v = min(v, max_value(gameState.result(a), alpha, beta, depth-1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(gameState, alpha, beta, depth):
            """ Return the value for a loss (-1) if the game is over,
            otherwise return the maximum value over all legal child
            nodes.
            """
            if gameState.terminal_test():
                return gameState.utility(self.player_id) 
            if depth <= 0:
                if heuristic_name == "custom":
                    return custom_heuristic(gameState)
                elif heuristic_name == "baseline":
                    return baseline_heuristic(gameState)
                elif heuristic_name == "greedy":
                    return greedy_heuristic(gameState)
                else:
                    return custom_heuristic(gameState) 

            v = float("-inf")
            for a in gameState.actions():
                v = max(v, min_value(gameState.result(a), alpha, beta, depth-1, heuristic_name))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def find_best_score(gameState):
            alpha = float("-inf")
            beta = float("inf")
            best_score = float("-inf")
            best_move = None
            for a in gameState.actions():
                v = min_value(gameState.result(a), alpha, beta, depth-1, heuristic_name)
                alpha = max(alpha, v)
                if v >= best_score:
                    best_score = v
                    best_move = a
            return best_move

        return find_best_score(state)

    def get_action(self, state): 
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # DONE: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        if state.ply_count <= 2: 
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.alpha_beta_search(state, depth=3, heuristic_name="custom")) # "custom", "baseline", or "greedy"
        
        # If need to debug, uncomment the following:
        #print('In get_action(), state received:')
        #debug_board = DebugState.from_state(state)
        #print(debug_board)