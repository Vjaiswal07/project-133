import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = []

with open("stars.csv",'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        data.append(row)

headers = data[0]
star_data = data[1:]

dataframe = pd.read_csv("final.csv")

stars_mass = dataframe["solar_mass"].tolist()
stars_radius = dataframe["solar_radius"].tolist()
star_names = dataframe["star_names"].tolist()

stars_mass.pop(0)
stars_radius.pop(0)
star_names.pop(0)

solar_mass = []

for row in solar_mass:
    si_unit = float(data)*1.989e+30
    solar_mass.append(si_unit)

solar_radius = []

for data in solar_radius:
    si_unit = float(data)* 6.957e+8
    solar_radius.append(si_unit)


star_mass = solar_mass
star_radius = solar_radius

star_gravities = []
for index,data in enumerate(star_names):
    gravity = (float(star_mass[index])*5.972e+24) / (float(star_radius[index])*float(star_radius[index])*6371000*6371000) * 6.674e-11
    star_gravities.append(gravity)

mass_list = dataframe["mass"].tolist()
mass_list.pop(0)

radius_list = dataframe["radius"].tolist()
radius_list.pop(0)

mass_radius_column = star_data.iloc[:, 2:4].values

within_cluster_sum_of_squares = []
for k in range(1, 9):
    k_means = KMeans(n_clusters = k, random_state = 42)
    k_means.fit(mass_radius_column)
    within_cluster_sum_of_squares.append(k_means.inertia_)

plt.figure(figsize = (10, 5))
sns.lineplot(x = range(1, 9), y = within_cluster_sum_of_squares, markers = 'bx-')

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of Squares')

k_means = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
prediction = k_means.fit_predict(mass_radius_column)

plt.figure(figsize = (10, 5))
sns.scatterplot(x = mass_radius_column[prediction == 0, 0], y = mass_radius_column[prediction == 0, 1], color = 'orange', label = 'Star Cluster 1')
sns.scatterplot(x = mass_radius_column[prediction == 1, 0], y = mass_radius_column[prediction == 1, 1], color = 'blue', label = 'Star Cluster 2')
sns.scatterplot(x = mass_radius_column[prediction == 2, 0], y = mass_radius_column[prediction == 2, 1], color = 'green', label = 'Star Cluster 3')
sns.scatterplot(x = k_means.cluster_centers_[:, 0], y = k_means.cluster_centers_[:, 1], color = 'red', label = 'Centroids', s = 100, marker = ',')

plt.title('Clusters of Stars')
plt.xlabel('Mass of Stars')
plt.ylabel('Radius of Stars')
plt.legend()
sns.scatterplot(x = mass_list,y = radius_list)
plt.title("STAR MASS AND RADIUS")
plt.xlabel('MASS')
plt.ylabel('RADIUS')
plt.show()
plt.figure()