import pandas as pd
import numpy as np
import itertools
import json, csv, os, sys
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
class ChartQA:
    def __init__(self, data_path='/home/yerong2/data/ChartQA/val/val_augmented.json'):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        return data
    
    def evaluate(self, ignore_keys=None, sanity_check=False, **kwargs):
        # Implement the evaluation logic specific to ChartQA here
        print(f"Evaluating ChartQA dataset with {len(self.data)} entries.")
        if sanity_check:
            print("Performing a sanity check for ChartQA.")
        # Example evaluation logic (use ignore_keys and kwargs as needed)
        return {"accuracy": 0.85}  # Dummy value

class PlotQA:
    def __init__(self, data):
        self.data = data
    
    def evaluate(self, ignore_keys=None, sanity_check=False, **kwargs):
        # Implement the evaluation logic specific to PlotQA here
        print(f"Evaluating PlotQA dataset with {len(self.data)} entries.")
        if sanity_check:
            print("Performing a sanity check for PlotQA.")
        # Example evaluation logic (use ignore_keys and kwargs as needed)
        return {"accuracy": 0.90}  # Dummy value

