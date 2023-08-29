import os

import lief
import lief.logging
import numpy
from torch.utils import data
from tqdm import tqdm

lief.logging.set_level(lief.logging.LOGGING_LEVEL.CRITICAL)

class FunctionIdentificationDataset(data.Dataset):
    """Generates the dataset for processing by the CNN"""
    def __init__(self, root):
        self._preprocess(root)
        
        
    def _preprocess(self, root):
        bins_paths = []
        bins_data = []
        bins_tags = []

        all_files = [os.path.join(dirpath, file) 
            for dirpath, _, filenames in os.walk(root) 
            for file in filenames]

        for file_path in tqdm(all_files):
            bin = lief.parse(file_path)
            if not bin:
                continue
            data = self._get_code(bin)

    def _get_code(self, bin):
        return numpy.array(bin.get_section(".text").content, dtype=int)