import argparse

WANDB_PROJECT = "CS6910_AS1"
WANDB_ENTITY = "ed23s037"

network_config = {}

parser = argparse.ArgumentParser()

def train():
    pass

args = parser.parse_args()
network_config.update(vars(args))