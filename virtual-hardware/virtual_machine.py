from components.board import VirtualBoard
class VirtualSNNSimulationMachine():
    def __init__(self, configuration):
        self.configuration = configuration
        self.cnt_boards = configuration.get("n_boards", 1)
        
        self._initialize_boards(configuration.get('board_configs', {
            'n_cores': 1,
            'core_configs':{
                
            }
        }))
    
    def _initialize_boards(self, board_configurations):
        cnt_boards = self.cnt_boards
        self.boards = []
        core_configuration
        for _ in range(cnt_boards):
            self.boards.append(VirtualBoard())
            
    def _initialize_board(self, board_configurations):
        return VirtualBoard()
        
    def _initialize_chips(self, board_configurations):
        return VirtualBoard()
            
    def _initialize_chip(self, board_configurations):
        return VirtualBoard()
    
    def _initialize_cores(self, board_configurations):
        return VirtualBoard()
    
    def _initialize_core(self, board_configurations):
        return VirtualBoard()
            