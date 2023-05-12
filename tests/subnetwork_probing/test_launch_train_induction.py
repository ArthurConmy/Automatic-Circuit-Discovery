#!/usr/bin/env python3

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "code"))

from launch_train_induction import main

def test_main():
    main(testing=True)
