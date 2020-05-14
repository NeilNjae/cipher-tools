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
from cipher.column_transposition import *
from cipher.vigenere import *

from support.text_prettify import *
from support.utilities import *
from support.plot_frequency_histogram import *
%matplotlib inline
```

```python
challenge_number = 7
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
fc = collections.Counter(sca)
plot_frequency_histogram(fc, sort_key=fc.get)
```

```python
kworda, score = vigenere_frequency_break(sca, fitness=Ptrigrams)
kworda
```

```python
ppa = vigenere_decipher(sca, kworda)
pa = repunctuate(ppa, pta)
print(pa)
```

```python
open(plaintext_a_filename, 'w').write(pa)
```

```python
fc = collections.Counter(scb)
plot_frequency_histogram(fc, sort_key=fc.get)
```

```python
kcb, score = caesar_break(scb, fitness=Pletters)
kcb
```

```python
ccb = caesar_decipher(scb, kcb)
ccb
```

```python
(kwordb, fillb, emptyb), score = column_transposition_break_mp(ccb, fitness=Ptrigrams)
(kwordb, fillb, emptyb), score
```

```python
pb = column_transposition_decipher(ccb, kwordb, fillcolumnwise=fillb, emptycolumnwise=emptyb)
pb
```

```python
fpb = lcat(tpack(segment(pb)))
print(fpb)
```

```python
open(plaintext_b_filename, 'w').write(fpb)
```

```python
transpositions[kwordb]
```

```python

```
