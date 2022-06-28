'''
 # @ Author: Rpeng
 # @ Create Time: 2022-06-12 22:49:32
 # @ Modified by: Rpeng
 # @ Modified time: 2022-06-16 22:49:48
 '''

from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly
import pandas as pd




pca = PCA(n_components= 3)

all_train_feature = np.load('train_feature.npy')
all_train_label = np.load('all_label.npy').reshape(-1,1)


all_test_feature = np.load('./processed_data/test_data/train_feature.npy')
all_test_label = np.load('./processed_data/test_data/all_label.npy').reshape(-1,1)

print(all_test_feature.shape)

pca.fit(all_train_feature)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_)


x_new = pca.transform(all_train_feature)

pca.fit(all_test_feature)

test_new = pca.transform(all_test_feature)

data = np.concatenate([x_new, all_train_label], axis = 1)

df = pd.DataFrame(data)

print(df.head())

label_0 = (all_train_label == 0).reshape(-1)
label_1 = (all_train_label == 1).reshape(-1)
label_2 = (all_train_label == 2).reshape(-1)
label_3 = (all_train_label == 3).reshape(-1)

# label_4 = (all_test_label == 0).reshape(-1)
# label_5 = (all_test_label == 1).reshape(-1)


# fig = px.scatter_3d(df, x = 0, y = 1, z= 2, color = 3, opacity=0.7, size = 5)

# plotly.offline.plot(fig, filename='3d_pca.html')

# trace = go.Scatter3d(
#     x = data[:, 0],
#     y = data[:, 1],
#     z = data[:,2],
#     mode = 'markers',
#     marker= dict(
#         size = 2,
#         color = data[:, 3]
#     )
# )

# fig = go.Figure([trace])

# plotly.offline.plot(fig, filename='3d_pca.html')



# trace0 = go.Scatter(
#     x = x_new[label_0][:, 0], 
#     y = x_new[label_0][:, 1], 
#     mode='markers',
#     marker=dict(
#         size = 5
#     ))

# trace1 = go.Scatter(
#     x = x_new[label_1][:, 0], 
#     y = x_new[label_1][:, 1], 
#     mode='markers',
#     marker=dict(
#         size = 5
#     ))

# trace2 = go.Scatter(
#     x = x_new[label_2][:, 0], 
#     y = x_new[label_2][:, 1], 
#     mode='markers',
#     marker=dict(
#         size = 5
# ))

# trace3 = go.Scatter(
#     x = x_new[label_3][:, 0], 
#     y = x_new[label_3][:, 1], 
#     mode='markers',
#     marker=dict(
#         size = 5
#     ))

# fig = go.Figure([trace0, trace1, trace2, trace3])

# plotly.offline.plot(fig, filename='2d_4_pca.html')



trace0 = go.Scatter3d(
    x = x_new[label_0][:, 0], 
    y = x_new[label_0][:, 1], 
    z = x_new[label_0][:, 2], 
    mode='markers',
    marker=dict(
        size = 2
    ))

trace1 = go.Scatter3d(
    x = x_new[label_1][:, 0], 
    y = x_new[label_1][:, 1], 
    z = x_new[label_1][:, 2], 
    mode='markers',
    marker=dict(
        size = 2
))

trace2 = go.Scatter3d(
    x = x_new[label_2][:, 0], 
    y = x_new[label_2][:, 1], 
    z = x_new[label_2][:, 2], 
    mode='markers',
    marker=dict(
        size = 2
))

trace3 = go.Scatter3d(
    x = x_new[label_3][:, 0], 
    y = x_new[label_3][:, 1], 
    z = x_new[label_3][:, 2], 
    mode='markers',
    marker=dict(
        size = 2
))

# trace4 = go.Scatter3d(
#     x = test_new[label_4][:, 0], 
#     y = test_new[label_4][:, 1],
#     z = test_new[label_4][:, 2],  
#     mode='markers',
#     marker=dict(
#         size = 2
#     ))

# trace5 = go.Scatter3d(
#     x = test_new[label_5][:, 0], 
#     y = test_new[label_5][:, 1],
#     z = test_new[label_5][:, 2],  
#     mode='markers',
#     marker=dict(
#         size = 2
#     ))


# trace3 = go.Scatter3d(
#     x = x_new[label_3][:, 0], 
#     y = x_new[label_3][:, 1], 
#     z = x_new[label_3][:, 2], 
#     mode='markers',
#     marker=dict(
#         size = 2
#     ))

fig = go.Figure([trace0, trace1, trace2, trace3])

plotly.offline.plot(fig, filename='512dim_3d_pca_train_test.html')