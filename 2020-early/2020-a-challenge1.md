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
```

```python
challenge_number = 1
plaintext_a_filename = f'{challenge_number}a.plaintext'
plaintext_b_filename = f'{challenge_number}b.plaintext'
ciphertext_a_filename = f'{challenge_number}a.ciphertext'
ciphertext_b_filename = f'{challenge_number}b.ciphertext'
```

```python
ca = open(ciphertext_a_filename).read()
cb = open(ciphertext_b_filename).read()

```

```python
k_a, score_a = caesar_break(ca)
print(k_a, '\n')
pa = caesar_decipher(ca, k_a)
print(pa)
```

```python
open(plaintext_a_filename, 'w').write(pa)
```

```python
k_b, score_b = caesar_break(cb)
print(k_b, '\n')
pb = caesar_decipher(cb, k_b)
print(pb)
```

```python
open(plaintext_b_filename, 'w').write(pb)
```

```python

```
