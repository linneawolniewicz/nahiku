# Third-party notices

## Balmung

The `src/nahiku/balmung.py` implementation was developed by Dan Hey and is
included in the source distribution and wheel with its original attribution.
It supplies the default prewhitening algorithm, retaining the collaborator-provided
fivefold frequency sampling, joint nonlinear sinusoid fitting, and fixed initial
amplitude threshold. Its numerical source is tracked against the copy in commit
`dac885e1f48100572a843e639b15a80f8b06b834`; a frozen test reference records that provenance.
Local edits to the engine are parameter documentation/comments and one explicit
boundary correction: `find_highest_peak` returns `nu[imax]` (frequency), not
`p[imax]` (amplitude), at the edge of the spectrum.

Nāhiku's adapter validates inputs, shifts the time origin, preserves the continuum,
and constructs common diagnostic results. It does not intercept Balmung optimizer
failures or replace its component-fitting algorithm. Constant flux requires no removal.
The alternative `method="weighted"` is Nāhiku's separate implementation using
Astropy LombScargle and bounded SciPy optimization; it is not presented as Balmung.

Upstream: https://github.com/danhey/balmung
License verified at https://github.com/danhey/balmung/blob/master/LICENSE on 2026-11-10.
The upstream copyright line is reproduced exactly below.

```text
MIT License

Copyright (c) 2018 Daniel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
