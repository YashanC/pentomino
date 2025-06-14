import json
import itertools
import random
import matplotlib.pyplot as plt

with open('assets/solutions2.json','r') as file:
    data = json.load(file)

jaccard = []
subset = random.sample(data,200)

for sol1,sol2 in itertools.combinations(subset,2):
    a_soln = set()
    b_soln = set()
    for i in range(len(sol1)):
        for coord in sol1[i]['coords']:
            a_soln.add(tuple(coord))
    for i in range(len(sol2)):
        for coord in sol2[i]['coords']:
            b_soln.add(tuple(coord))
    union = a_soln | b_soln
    intersect = a_soln & b_soln
    jaccard.append(len(intersect)/len(union))

print(sum(1 for x in jaccard if x>0.6)/len(jaccard))
plt.hist(jaccard,bins=20)
plt.show()
    
