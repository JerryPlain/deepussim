"""Learned US renderer (B1): CBCT slice -> US-like image via contrastive unpaired translation.

The physics renderer in :mod:`deepussim.us.renderer` stays as a structural baseline; this package
is the *learned* appearance model — a CUT generator trained against real US (unpaired), with the
adversarial + PatchNCE objective replacing the Nelder-Mead parameter fit. See ``docs/renderer.md``.
"""
