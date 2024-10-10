from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def generate_graphene_lattice(size):
    a1 = np.array([np.sqrt(3), 0, 0])
    a2 = np.array([np.sqrt(3)/2, 3/2, 0])

    points = []
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            point = i*a1 + j*a2
            points.append(point)
            points.append(point + np.array([0, 1, 0]))
    
    return np.array(points)

size = 4
graphene_points = generate_graphene_lattice(size)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(graphene_points[:,0], graphene_points[:,1], graphene_points[:,2], c='black')

for p in graphene_points:
    for q in graphene_points:
        if np.allclose(np.linalg.norm(p-q), 1, rtol=1e-05, atol=1e-08):
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c='black', linewidth=0)

ax.view_init(elev=20., azim=45)
plt.title('Graphene Lattice Structure in 3D')
plt.savefig('structure.png', bbox_inches='tight', dpi=300)
