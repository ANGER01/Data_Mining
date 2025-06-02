import pandas as pd
import numpy as np
import random

stuff: pd.DataFrame = pd.read_csv("M.csv", header=None)
stuff = stuff.to_numpy()


def matrix_power(m, q, t):
    m_star = np.linalg.matrix_power(m,t)
    
    return np.dot(m_star, q.T)


def state_prop(m, q, t):
    for i in range(0, t):
        q_star = np.dot(m, q.T)
        q = q_star
    
    return q


def random_walk(m: np.ndarray, t_: int, t: int):
    q = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    final = np.zeros(10)
    for i in range(0, t_):
        q = np.dot(m, q.T)
        nodes = []
        weights = []
        for j in range(0, len(q)):
            if q[j] > 0:
                nodes.append(j)
                weights.append(q[j])
        choisir = np.random.choice(nodes, replace=False, p=weights)
        q = np.zeros(10)
        q[choisir] = 1
        
    for i in range(0, t):
        if(np.random.choice(['a','b'], p=[0.15, .85])) != 'a':
            q = np.dot(m, q.T)
            nodes = []
            weights = []
            for j in range(0, len(q)):
                if q[j] > 0:
                    nodes.append(j)
                    weights.append(q[j])
            choisir = np.random.choice(nodes, p=weights)
            q = np.zeros(10)
            q[choisir] = 1
            final[choisir] += 1
        else:
            chois = random.randint(0,9)
            final[chois] += 1
            q = np.zeros(10)
            q[chois] = 1

    return final / np.sum(np.abs(final))


def eigen(m):
    _, vecs = np.linalg.eig(m)
    v = vecs[:,0].real
    return np.abs(v / np.sum(np.abs(v)))

q = np.full(10, .1)
#print("Matrix Power:", matrix_power(m=stuff, q=q, t=120), "\n")
#print("State Propagation:", state_prop(m=stuff, q=q, t=120), "\n")
print("Random Walk:", random_walk(stuff, 100, 1024), "\n")
#print("Eigen-Analysis:", eigen(m=stuff), "\n")