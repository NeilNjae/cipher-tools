---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
```

```python
from cipher.caesar import *
from cipher.affine import *
from cipher.keyword_cipher import *

from support.text_prettify import *
from support.utilities import *
from support.plot_frequency_histogram import *
```

```python
challenge_number = 3
plaintext_a_filename = f'{challenge_number}a.plaintext'
plaintext_b_filename = f'{challenge_number}b.plaintext'
ciphertext_a_filename = f'{challenge_number}a.ciphertext'
ciphertext_b_filename = f'{challenge_number}b.ciphertext'
```

```python
ca = open(ciphertext_a_filename).read()
cb = open(ciphertext_b_filename).read()

sca = sanitise(ca)
pca = letters(ca)
pta = depunctuate(ca)

scb = sanitise(cb)
pcb = letters(cb)
ptb = depunctuate(cb)
```

```python
(kworda, kwrapa), score = keyword_break_mp(sca, fitness=Ptrigrams)
kworda, kwrapa
```

```python
pa = keyword_decipher(ca, kworda, kwrapa)
print(pa)
```

```python
open(plaintext_a_filename, 'w').write(pa)
```

```python
(kwordb, kwrapb), score = keyword_break_mp(scb, fitness=Ptrigrams)
kwordb, kwrapb
```

```python
pb = keyword_decipher(cb, kwordb, kwrapb)
print(pb)
```

```python
open(plaintext_b_filename, 'w').write(pb)
```

```python

```
