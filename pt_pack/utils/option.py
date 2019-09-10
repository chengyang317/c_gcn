# coding=utf-8
import argparse


def add_argument(group: argparse.ArgumentParser, key, **kwargs):
    if isinstance(group, argparse.ArgumentParser):
        actions = group._actions
    else:
        actions = group._group_actions
    action_names = [action.dest for action in actions]
    if key not in action_names:
        group.add_argument(f'--{key}', **kwargs)
    return group