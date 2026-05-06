# std lib imports

# 3-party import
import pandas as pd

# projekt imports
from sciflow.data.scifate_dataset import ScifateDataLoader, ScifateDataset




scifate_loader = ScifateDataLoader()
scifate_data = scifate_loader.load(folder_path="/home/nicolas/code/master/higdim/HighDim/data/ScifateData")


print(scifate_data.summary())



# print(scifate_data.cell_info)

# print(scifate_data.expression_matrix)

print(scifate_data.expression_matrix.col_info)
print(scifate_data.expression_matrix.row_info)


# print(scifate_data.expression_matrix.to_labeled())