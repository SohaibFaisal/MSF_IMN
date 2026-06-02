import pandas as pd
import torch


def tangent_from_csv3d(fpath: str, *, csv_dict=None) -> torch.Tensor:
    r"""turn material tangent csv file to torch tensor
        input: csv of 9x9
        output: torch.tensor 6x6"""

    csv_dict = {
        'C0000': [0, 0],
        'C0011': [0, 1],
        'C0022': [0, 2],
        'C0001': [0, 3],
        'C0002': [0, 4],
        'C0012': [0, 5],
        'C1100': [1, 0],
        'C1111': [1, 1],
        'C1122': [1, 2],
        'C1101': [1, 3],
        'C1102': [1, 4],
        'C1112': [1, 5],
        'C2200': [2, 0],
        'C2211': [2, 1],
        'C2222': [2, 2],
        'C2201': [2, 3],
        'C2202': [2, 4],
        'C2212': [2, 5],
        'C0100': [3, 0],
        'C0111': [3, 1],
        'C0122': [3, 2],
        'C0101': [3, 3],
        'C0102': [3, 4],
        'C0112': [3, 5],
        'C0200': [4, 0],
        'C0211': [4, 1],
        'C0222': [4, 2],
        'C0201': [4, 3],
        'C0202': [4, 4],
        'C0212': [4, 5],
        'C1200': [5, 0],
        'C1211': [5, 1],
        'C1222': [5, 2],
        'C1201': [5, 3],
        'C1202': [5, 4],
        'C1212': [5, 5],
    }

    # read data
    try:
        df = pd.read_csv(fpath, sep=';')
    except pd.errors.ParserError:
        df = pd.read_csv(fpath, sep=',')

    CTan = torch.zeros(df.shape[0], 6, 6)

    for key, val in csv_dict.items():
        row, col = val
        CTan[:, row, col] = torch.from_numpy(df[key].values)

    return CTan



