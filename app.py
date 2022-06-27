from dataclasses import dataclass
from functools import reduce
from rtree import index

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 100*200
vmax = 0.03
d0 = 0.0125
dc = 0.01
l0 = 0.31
l1 = 0.001
l2 = 1.2
l3 = 2
l4 = 0.01


@dataclass
class Boid:
    position: np.ndarray = np.random.random(3)
    velocity: np.ndarray = np.array([0.0, 0.01, 0.01])


def distance(p1, p2) -> float:
    return np.linalg.norm(p1 - p2, 2)


# init
p = index.Property()
p.dimension = 3
idx = index.Index(properties=p)
boids = [Boid(np.random.random(3), np.random.random(3) / 100) for _ in range(n + 1)]

for i, b in enumerate(boids):
    idx.add(i,b.position.tolist())


def update(boid: Boid, other_b: [Boid], idx):
    hacky_other_b = [boids[bid] for bid in idx.nearest(boid.position.tolist(), 10)]
    others = [b for b in hacky_other_b if distance(b.position, boid.position) < do]
    others_close = [b for b in others if distance(b.position, boid.position) < dc]
    others_len = len(others)
    others_close_len = len(others_close)

    w1 = (1 / others_len * reduce(lambda x, y: x + y, [x.position for x in others])) - boid.position
    w2 = (1 / others_len * reduce(lambda x, y: x + y, [x.velocity for x in others]))
    w3 = (1 / others_close_len * reduce(lambda x, y: x + y, [o.position - boid.position for o in others_close]))
    w4 = np.zeros(3) if np.max(np.abs(boid.position)) <= 1 else -boid.position

    velocity = boid.velocity * l0 + w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4

    if np.linalg.norm(velocity) > vmax:
        velocity = vmax * (velocity / np.linalg.norm(velocity))

    position = boid.position + velocity
    return Boid(position, velocity)

"""
for t in range(20):
    print(t)
    boids = [update(b, boids, idx) for b in boids]
    for i, b in enumerate(boids):
        idx.add(i,b.position.tolist())
"""


#boids = [Boid(np.random.random(3), np.random.random(3) / 100) for _ in range(n + 1)]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
graph = ax.scatter([x.position[0] for x in boids], [x.position[1] for x in boids], [x.position[2] for x in boids])



def update_plt(num):
    print(num)
    global boids
    boids = [update(b, boids,idx) for b in boids]
    graph._offsets3d = (np.array([x.position[0] for x in boids]), np.array([x.position[1] for x in boids]),
                        np.array([x.position[2] for x in boids]))

    for i, b in enumerate(boids):
        idx.add(i,b.position.tolist())

    return graph,


ani = animation.FuncAnimation(fig, update_plt, frames=2000, blit=True)
ani.save("test.mp4", fps=30)
