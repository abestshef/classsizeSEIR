### Import the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from random import *

### Global constants

POPSIZE=1000 # Total individuals in population
NOHOUSES=100 # No. of households (mean household size = POPSIZE / NOHOUSES)

bigstore=[] # Used to store peaks
bigstore2=[] # Used to store totals
bigstorer0=[] # Used to store R0s

OMEGA=1/7 # Latent rate
GAMMA=1/14 # Recovery rate
BETA=np.zeros(2)
BETA[0]=0.2*GAMMA # Transmission coefficient at home
BETA[1]=0.2*GAMMA # Transmission coefficient at school

REPS=100 # No. of replicate runs
I0=25 # Initial no. of infected individuals (mean)
MAX_TIME=180 # Max time
CLASS_START = 0.4 # Time in day when move to class (SHOULD THESE BE FIXED OR VARIABLE?)
CLASS_END = 0.5 # Time in day when move to house

### Main code

### Functions

# Function to count numbers in each compartment
def count_type(type):
    return np.count_nonzero(PERSON[:,3] == type)

# Function to check the current scale
def findscale(home):
    # Check location
    loc = NOHOUSES if home==1 else NOCLASSES
    counts = np.zeros((loc, 3))
    for j in range(0, loc):
        for c in range(0, 3):
            counts[j, c] = np.count_nonzero(np.logical_and(PERSON[:,home]==j, PERSON[:,3]==c))  
    # Sum of all rates
    scale = np.sum(OMEGA*counts[:,1] + GAMMA*counts[:,2] + BETA[home-1]*counts[:,0]*counts[:,2])
    return scale

# Main function
for classes in range(1,11):
    NOCLASSES=int(POPSIZE/(classes*5)) # No. of classes (mean class size = POPSIZE / NOCLASSES)
    print(NOCLASSES)
    peaks=[] # For storing peak no. of infecteds
    tots=[] # For storing total no. infected
    R0store=np.zeros(REPS)
    
    for reps in range(0,REPS):

        # Assign population to houses/classes/sir
        PERSON=np.zeros((POPSIZE,4))
        # 0th col: ID 
        # 1st col: house
        # 2nd col: class
        # 3rd col: SIR status
        infs=np.sort(np.random.choice(POPSIZE,I0))
        for i in range(0,POPSIZE):
            PERSON[i][0] = i
            PERSON[i][1] = randint(0,NOHOUSES-1)
            PERSON[i][2] = randint(0,NOCLASSES-1)
            if i in infs:
                PERSON[i][3] = 2 # Initially everyone susceptible except I0 individuals

        # Some local constants / lists
        tsteps=[0]
        infecteds=[count_type(2)]
        susceptibles=[count_type(0)]
        exposed=[count_type(1)]
        current_t=0  
        home=1 # Everyone starts at home

        # Main run
        while current_t < MAX_TIME and np.any(PERSON[:,3] == 2):

            # Find proposed time to next event
            scale = findscale(home)
            dt = -np.log(np.random.uniform()) / scale
            proposed_t = tsteps[-1] + dt
            
            if home == 1 and proposed_t > int(proposed_t) + CLASS_START and proposed_t < int(proposed_t) + CLASS_END:
                # If students are home and proposed time of next event is later 
                # than class starts, then no event occurs before class starts
                current_t = int(proposed_t)+CLASS_START
                home = 2
            elif home == 2 and proposed_t > int(proposed_t) + CLASS_END:
                # If students are in class and proposed time of next event is 
                # later than class ends, then no event occurs before class ends
                current_t = int(proposed_t)+CLASS_END
                home = 1
            else:
                # Next event occurs before class starts/ends
                current_t = proposed_t

                # Find event
                eventcheck = np.random.uniform()
                if eventcheck < GAMMA*infecteds[-1]/scale: #Event is recovery  
                    
                    # If there are any infected people, randomly choose one to recover
                    infected_indices = np.where(PERSON[:,3] == 2)
                    if infected_indices:
                        PERSON[np.random.choice(infected_indices[0]), 3] = 3

                elif eventcheck < (GAMMA*infecteds[-1] + OMEGA*exposed[-1])/scale: # Event is latent->infected

                    # If there are any latents, randomly choose one to become infected
                    latent_indices = np.where(PERSON[:,3] == 1)
                    if latent_indices:
                        PERSON[np.random.choice(latent_indices[0]), 3] = 2

                else: #Event is transmission
                    findx=[i for i in range(POPSIZE)] # Make a randomised list of everyone to ensure individuals chosen randomly
                    shuffle(findx)
                    for tryx in findx:
                        if PERSON[tryx,3]==0: # Find a susceptible host
                            loc=PERSON[tryx,home]
                            contacts=np.where(PERSON[:,home]==loc)
                            infcontacts=np.where(PERSON[contacts[0],3]==2)
                            if infcontacts[0].size>0:
                                PERSON[tryx,3]=1
                                # Check if founder made infection, if so add to R0
                                currentinfs=np.where(PERSON[infs,3]==2)
                                currentinfs_ind=(infs[currentinfs[0]])
                                if currentinfs[0].size>0:
                                    idx = np.searchsorted(currentinfs_ind,contacts[0])
                                    idx[idx==len(currentinfs_ind)] = 0
                                    mask = currentinfs_ind[idx]==contacts[0]
                                    if np.random.uniform()<sum(mask)/infcontacts[0].size:
                                        R0store[reps]+=1/I0
                                break                         

            # Update lists
            tsteps.append(current_t)
            infecteds.append(count_type(2))
            exposed.append(count_type(1))

            # Stop if infections has finished
            if infecteds[-1]==0:
                break

        # Find peak no. infected
        peaks.append(max(infecteds)/POPSIZE)
        tots.append((count_type(2)+count_type(3))/POPSIZE)

    print("The median peak is", np.median(peaks))
    print("The median R0 is", np.median(R0store))
    
    bigstore.append(peaks)
    bigstore2.append(tots)
    bigstorer0.append(R0store)
         
### Plotting code
    
xx=[j*5 for j in range (1,11)]
plt.rcParams.update({'font.size': 16})

# Peaks boxplot
fig, ax = plt.subplots()
data=[bigstore[:][0],bigstore[:][1],bigstore[:][2],bigstore[:][3],bigstore[:][4],bigstore[:][5],bigstore[:][6],bigstore[:][7],bigstore[:][8],bigstore[:][9]]
plt.boxplot(data)
ax.set(xticklabels=xx)
plt.xlabel('Average class size')
plt.ylabel('Peak infected')
plt.ylim(0,0.4)
plt.tight_layout()
plt.savefig('peaksbox.png')

# Totals boxplot
fig, ax = plt.subplots()
data=[bigstore2[:][0],bigstore2[:][1],bigstore2[:][2],bigstore2[:][3],bigstore2[:][4],bigstore2[:][5],bigstore2[:][6],bigstore2[:][7],bigstore2[:][8],bigstore2[:][9]]
plt.boxplot(data)
ax.set(xticklabels=xx)
plt.xlabel('Average class size')
plt.ylabel('Total infected')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('totsbox.png')

# R0s boxplot
fig, ax = plt.subplots()
data=[bigstorer0[:][0],bigstorer0[:][1],bigstorer0[:][2],bigstorer0[:][3],bigstorer0[:][4],bigstorer0[:][5],bigstorer0[:][6],bigstorer0[:][7],bigstorer0[:][8],bigstorer0[:][9]]
plt.boxplot(data)
ax.set(xticklabels=xx)
plt.xlabel('Average class size')
plt.ylabel('Reproductive ratio, $R_0$')
plt.ylim(0,5)
plt.tight_layout()
plt.savefig('r0sbox.png')

r0q=np.zeros((10,4))
totsq=np.zeros((10,4))
peaksq=np.zeros((10,4))
for i in range(10):
    r0q[i,0]=xx[i]
    totsq[i,0]=xx[i]
    peaksq[i,0]=xx[i]
    for j in range(3):
        r0q[i,j+1]=np.quantile(bigstorer0[:][i],(j+1)*0.25)
        totsq[i,j+1]=np.quantile(bigstore2[:][i],(j+1)*0.25)
        peaksq[i,j+1]=np.quantile(bigstore[:][i],(j+1)*0.25)

np.savetxt("R0quartiles.txt",r0q,fmt="%.3f")
np.savetxt("totsquartiles.txt",totsq,fmt="%.3f")
np.savetxt("peaksquartiles.txt",peaksq,fmt="%.3f")

np.savetxt("R0store10002.txt",bigstorer0,fmt="%.3f")
np.savetxt("totstore10002.txt",bigstore2,fmt="%.3f")
np.savetxt("peakstore10002.txt",bigstore,fmt="%.3f")