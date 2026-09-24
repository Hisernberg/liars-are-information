"""LAI: "Liars Are Information" v3 -- receiver-anchored channel estimation.

This package holds the v3 contributions built on top of the original ``aip``
package (which is kept unchanged as the baseline implementation):

* :mod:`lai.race` -- RACE, Receiver-Anchored Channel Estimation. A label-free
  latent-truth model fitted by each honest receiver on its own history, anchored
  on the one fact the receiver knows (that it is itself honest). It replaces
  AIP's TRUST / DISCARD / INVERT gate with a continuous signed log-odds weight.
* :mod:`lai.data` -- cache loading, gold-free answer canonicalisation, scoring.
* :mod:`lai.attacks` -- symbolic and replayed-LLM Byzantine strategies,
  including strategies built to defeat RACE specifically.
* :mod:`lai.sim` -- world construction and the evaluation harness.
* :mod:`lai.stats` -- paired task-cluster bootstrap and multiplicity control.
* :mod:`lai.theory` -- the information bound used as an upper reference.
"""

__version__ = "3.0.0"
