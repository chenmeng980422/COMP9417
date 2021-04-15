import numpy as np

data=np.array(df)
data_x=data[:,range(3)]
data_y=data[:,3]
ones_data=np.ones([len(data),1])
data_x_ones=np.column_stack((ones_data,data_x))