# API reference

The public interface is exported from `nahiku`. Preprocessing and fitting methods
that support chaining return the `Nahiku` object. Prewhitening and searches return
separate result objects. Plotting returns Matplotlib figures and axes without
calling `show()`.

```{eval-rst}
.. autoclass:: nahiku.Nahiku
   :members:
   :undoc-members:

.. autoclass:: nahiku.SearchResult
   :members:

.. autoclass:: nahiku.PrewhiteningResult
   :members:

.. autoclass:: nahiku.QuasiPeriodicKernel
   :members:
```

Internal modules are implementation details outside the supported public interface. See `examples/` for complete workflows and `tests/` for
numerical validation.
