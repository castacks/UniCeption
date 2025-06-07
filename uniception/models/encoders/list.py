"""
List available UniCeption encoders.
"""

import argparse

from uniception.models.encoders import print_available_encoder_models

if __name__ == "__main__":
    print_available_encoder_models()
