import string
import collections
import multiprocessing
import itertools
import logging

from cipher.enigma import *


logger = logging.getLogger('bombe')
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# create the logging file handler
fh = logging.FileHandler("enigma.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add handler to logger object
logger.addHandler(fh)

##################################
# # Bombe
##################################
#
# Good explanation of [how the bombe worked](http://www.ellsbury.com/enigmabombe.htm) by Graham Ellsbury
#

Signal = collections.namedtuple('Signal', ['bank', 'wire'])
Connection = collections.namedtuple('Connection', ['banks', 'scrambler'])
MenuItem = collections.namedtuple('MenuIem', ['before', 'after', 'number'])


def make_menu(plaintext, ciphertext):
    return [MenuItem(p, c, i+1) 
            for i, (p, c) in enumerate(zip(plaintext, ciphertext))]


class Scrambler(object):
    def __init__(self, wheel1_spec, wheel2_spec, wheel3_spec, reflector_spec,
                 wheel1_pos='a', wheel2_pos='a', wheel3_pos='a'):
        self.wheel1 = SimpleWheel(wheel1_spec, position=wheel1_pos)
        self.wheel2 = SimpleWheel(wheel2_spec, position=wheel2_pos)
        self.wheel3 = SimpleWheel(wheel3_spec, position=wheel3_pos)
        self.reflector = Reflector(reflector_spec)
    
    def __getattribute__(self, name):
        if name=='wheel_positions':
            return self.wheel1.position, self.wheel2.position, self.wheel3.position 
        elif name=='wheel_positions_l':
            return self.wheel1.position_l, self.wheel2.position_l, self.wheel3.position_l 
        else:
            return object.__getattribute__(self, name)
    
    def advance(self, wheel1=False, wheel2=False, wheel3=True):
        if wheel1: self.wheel1.advance()
        if wheel2: self.wheel2.advance()
        if wheel3: self.wheel3.advance()
            
    def lookup(self, letter):
        a = self.wheel3.forward(letter)
        b = self.wheel2.forward(a)
        c = self.wheel1.forward(b)
        d = self.reflector.forward(c)
        e = self.wheel1.backward(d)
        f = self.wheel2.backward(e)
        g = self.wheel3.backward(f)
        return g
    
    def set_positions(self, wheel1_pos, wheel2_pos, wheel3_pos):
        self.wheel1.set_position(wheel1_pos)
        self.wheel2.set_position(wheel2_pos)
        self.wheel3.set_position(wheel3_pos)        


class Bombe(object):
    
    def __init__(self, wheel1_spec, wheel2_spec, wheel3_spec, reflector_spec,
                menu=None, start_signal=None, use_diagonal_board=True, 
                verify_plugboard=True):
        self.connections = []
        self.wheel1_spec = wheel1_spec
        self.wheel2_spec = wheel2_spec
        self.wheel3_spec = wheel3_spec
        self.reflector_spec = reflector_spec
        if menu:
            self.read_menu(menu)
        if start_signal:
            self.test_start = start_signal
        self.use_diagonal_board = use_diagonal_board
        self.verify_plugboard = verify_plugboard
        
    def __getattribute__(self, name):
        if name=='wheel_positions':
            return self.connections[0].scrambler.wheel_positions
        elif name=='wheel_positions_l':
            return self.connections[0].scrambler.wheel_positions_l
        else:
            return object.__getattribute__(self, name)
        
    def __call__(self, start_positions):
        return start_positions, self.test(initial_signal=self.test_start,
            start_positions=start_positions, 
            use_diagonal_board=self.use_diagonal_board,
            verify_plugboard=self.verify_plugboard)
        
    def add_connection(self, bank_before, bank_after, scrambler):
        self.connections += [Connection([bank_before, bank_after], scrambler)]
        
    def read_menu(self, menu):
        self.connections = []
        for item in menu:
            scrambler = Scrambler(self.wheel1_spec, self.wheel2_spec, self.wheel3_spec,
                                  self.reflector_spec,
                                  wheel3_pos=unpos(item.number - 1))
            self.add_connection(item.before, item.after, scrambler)
        most_common_letter = (collections.Counter(m.before for m in menu) +\
            collections.Counter(m.after for m in menu)).most_common(1)[0][0]
        self.test_start = Signal(most_common_letter, most_common_letter)
        
    def set_positions(self, wheel1_pos, wheel2_pos, wheel3_pos):
        for i, c in enumerate(self.connections):
            c.scrambler.set_positions(wheel1_pos, wheel2_pos, unpos(pos(wheel3_pos) + i))
    
    def test(self, initial_signal=None, start_positions=None, use_diagonal_board=True,
            verify_plugboard=True):
        self.banks = {label: 
                      dict(zip(string.ascii_lowercase, [False]*len(string.ascii_lowercase)))
                      for label in string.ascii_lowercase}
        if start_positions:
            self.set_positions(*start_positions)
        if not initial_signal:
            initial_signal = self.test_start
        self.pending = [initial_signal]
        self.propagate(use_diagonal_board)
        live_wire_count = len([self.banks[self.test_start.bank][w] 
                    for w in self.banks[self.test_start.bank] 
                    if self.banks[self.test_start.bank][w]])
        if live_wire_count < 26:
            if verify_plugboard:
                possibles = self.possible_plugboards()
                return all(s0.isdisjoint(s1) for s0 in possibles for s1 in possibles if s0 != s1)
            else:
                return True
        else:
            return False
        
    def propagate(self, use_diagonal_board):
        while self.pending:
            current = self.pending[0]
            # print("processing", current)
            logger.debug("Propogater processing {}".format(current))
            self.pending = self.pending[1:]
            if not self.banks[current.bank][current.wire]:
                self.banks[current.bank][current.wire] = True
                if use_diagonal_board:
                    self.pending += [Signal(current.wire, current.bank)]
                for c in self.connections:
                    if current.bank in c.banks:
                        other_bank = [b for b in c.banks if b != current.bank][0]
                        other_wire = c.scrambler.lookup(current.wire)
                        # print("  adding", other_bank, other_wire, "because", c.banks)
                        logger.debug("Propogator adding {0} {1} because {2}".format(other_bank, other_wire, c.banks))
                        self.pending += [Signal(other_bank, other_wire)]
    
    def run(self, run_start=None, wheel1_pos='a', wheel2_pos='a', wheel3_pos='a', use_diagonal_board=True):
        if not run_start:
            run_start = self.test_start
        self.solutions = []
        self.set_positions(wheel1_pos, wheel2_pos, wheel3_pos)
        for run_index in range(26*26*26):
            if self.test(initial_signal=run_start, use_diagonal_board=use_diagonal_board):
                self.solutions += [self.connections[0].scrambler.wheel_positions_l]
            advance3 = True
            advance2 = False
            advance1 = False
            if (run_index + 1) % 26 == 0: advance2 = True
            if (run_index + 1) % (26*26) == 0: advance1 = True
            for c in self.connections:
                c.scrambler.advance(advance1, advance2, advance3)
        return self.solutions
    
    def possible_plugboards(self):
        possibles = set()
        for b in self.banks:
            active = [w for w in self.banks[b] if self.banks[b][w]]
            inactive = [w for w in self.banks[b] if not self.banks[b][w]]
            if len(active) == 1:
                possibles = possibles.union({frozenset((b, active[0]))})
            if len(inactive) == 1:
                possibles = possibles.union({frozenset((b, inactive[0]))})
        return possibles


def run_multi_bombe(wheel1_spec, wheel2_spec, wheel3_spec, reflector_spec, menu,
                    start_signal=None, use_diagonal_board=True, 
                    verify_plugboard=True):
    allwheels = itertools.product(string.ascii_lowercase, repeat=3)

    with multiprocessing.Pool() as pool:
        res = pool.map(Bombe(wheel1_spec, wheel2_spec, wheel3_spec, 
            reflector_spec, menu=menu, start_signal=start_signal, 
            use_diagonal_board=use_diagonal_board, 
            verify_plugboard=verify_plugboard),
                  allwheels)
    return [r[0] for r in res if r[1]]