"""Reconstruct all FNR artifacts from the public 2016–2022 raw tables."""
if __package__:
    from .fnr.training.pipeline import main
else:
    from fnr.training.pipeline import main

if __name__ == "__main__":
    main()
