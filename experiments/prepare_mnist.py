#!/usr/bin/env python3
"""Write .data/mnist.npz (x_train, y_train, x_test, y_test; uint8) for the MNIST scripts.

Uses torchvision's MNIST download, so no extra dependency is needed.
Run from the experiments/ directory:  python prepare_mnist.py
"""
import os

import numpy as np
from torchvision import datasets


def main(root=".data"):
    os.makedirs(root, exist_ok=True)
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    out = os.path.join(root, "mnist.npz")
    np.savez_compressed(out,
                        x_train=tr.data.numpy().astype(np.uint8),
                        y_train=tr.targets.numpy().astype(np.int64),
                        x_test=te.data.numpy().astype(np.uint8),
                        y_test=te.targets.numpy().astype(np.int64))
    print(f"wrote {out}: train {tr.data.shape}, test {te.data.shape}")


if __name__ == "__main__":
    main()
