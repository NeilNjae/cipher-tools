import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from cipher.caesar import *
from support.text_prettify import *

c1a = open('1a.ciphertext').read()
c1b = open('1b.ciphertext').read()

key_a, score_a = caesar_break(c1a)
open('1a.plaintext', 'w').write(caesar_decipher(c1a, key_a))

key_b, score_b = caesar_break(c1b)
open('1b.plaintext', 'w').write(caesar_decipher(c1b, key_b))
