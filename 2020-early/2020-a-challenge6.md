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

from support.text_prettify import *
from support.utilities import *
from support.plot_frequency_histogram import *
%matplotlib inline
```

```python
challenge_number = 6
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
(kworda, kwrapa), score = keyword_break_mp(sca, fitness=Ptrigrams)
kworda, kwrapa
```

```python
pa = cat(reversed(keyword_decipher(sca, kworda, kwrapa)))
print(pa)
```

```python
fpa = lcat(tpack(segment(pa)))
print(fpa)
```

```python
open(plaintext_a_filename, 'w').write(fpa)
```

```python
fc = collections.Counter(scb)
plot_frequency_histogram(fc, sort_key=fc.get)
```

```python
rscb = cat(reversed(cb))
rscb
```

```python
(kwordb, fillb, emptyb), score = column_transposition_break_mp(scb, fitness=Ptrigrams)
(kwordb, fillb, emptyb), score
```

```python
pb = column_transposition_decipher(scb, kwordb, fillcolumnwise=fillb, emptycolumnwise=emptyb)
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
