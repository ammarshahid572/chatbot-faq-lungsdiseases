#!/usr/bin/env python
# coding: utf-8

# In[11]:


mainQuery= {
    "greeting": ["Hi", "Hello", "How are you?"],
    "covid": ["covid", "covid-19", "covid19", "sars-cov"],
    "tuberculosis": ['tuberculosis', 'tb'],
    "asthma" : ["asthma"],
    "nocontext": [""],
    "endNote": ["Thank you", "This was helpful!", "See you later!"]
}
context1 = {
    "diagnosis":["if i have", "if he has", "if she has", "if they have", "diagnosis"],
    "medications": ["treatment", "medications", "medicine"],
    "types":["types", "variants", "stages"],
    "facts":["what is that?","tell me", "when did", "what happens"],
    "nocontext":[""]
}

context2= {
    "sideeffects":["side effects", "careful", "effects"],
    "preparation":["preparation", "prepare for", "requirements"],
    "duration": ["how long", "duration"],
    "nocontext":[""]
}

actions= {
    "greeting": ["Hello"],
    "asthmaDef": ["Asthma Definition"],
    "covidDef": ["Covid Definition"],
    "tbDef": ["Covid Definition"],
    "endNote": ["Thanks for chatting"],
    "covidDiag":["PCR Test"],
    "asthmaDiag":["Asthma Diagnosis"],
    "tbDiag":["Tb diagnosis"],
    "noContext": ["Sorry didnt Understand the question. please specify in details"],
    "covidvariants": ["covid variants"],
    "ashtmavariants": ["asthma variants"],
    "tbvariants":["Tb variants"],
    "covidMeds": ["covidMedications"],
    "tbMeds": ["Tb meds"],
    "asthmaMeds":["asthma meds"],
    "unrelated": ["sorry I can only answer for question I am allowed to."]

}


# In[12]:


def getContext(text):
    diseaseaseContext="nocontext"
    contexta="nocontext"
    contextb="nocontext"
    for key in mainQuery:
        for keyword in mainQuery[key]:
            if keyword in text:
                diseaseaseContext=key
                break
        else:
            continue
        break
    for key in context1:
        for keyword in context1[key]:
            if keyword in text:
                contexta=key
                break
        else:
            continue
        break
    for key in context2:
        for keyword in context2[key]:
            if keyword in text:
                contextb=key
                break
        else:
            continue
        break
    return(diseaseaseContext, contexta, contextb)


# In[13]:


states= list()
for key in sorted(mainQuery):
    for context in sorted(context1):
        for  intent in sorted(context2):
            states.append((key,context, intent))
print(states[7])


# In[14]:


print (len(states))
print (len(actions))



# In[15]:


import numpy as np

#set up  mapping and inverse mapping to find corresponeding state or number
numbertoState=dict()
statetoNumber=dict()
for i,state in enumerate(states):
    numbertoState[i]= state
    statetoNumber[state]=i
numbertoAction=dict()
actiontoNumber=dict()
for i,action in enumerate(actions):
    numbertoAction[i]= action
    actiontoNumber[action]=i



# In[16]:


#reward map----- old state to new state reward
reward= np.zeros((len(states), len(states)),dtype=float)
print(reward.shape)
# since we know we have to somehow end the convo. lets set maximum reward if the next state is ending the convo
for i,lastState in enumerate(states):
    for j,nextState in enumerate(states):
        if "endNote" in nextState:
            reward[i][j] =1


# In[17]:


qTable= np.zeros((len(states), len(actions)), dtype=float)
with open('test.npy','rb')as f:
    f.seek(0)
    loadedFile=np.load(f ,allow_pickle=False)
    if loadedFile.shape== qTable.shape:
        
        qTable=loadedFile
        print("Loaded old backup")


# Qtable Training

# In[19]:


import random
#start q learn algo
# Initialize parameters
gamma = 0.75 # Discount factor 
alpha = 0.9 # Learning rate 
epsilon = 0.1
next_stateIndex=0
stateIndex=0
actionNumber=0
episodeRun=True
reward=0
prevState=0
while episodeRun:
    text=input("Text:")
    state=getContext(text)
    stateIndex=statetoNumber[state]

    next_max = np.max(qTable[stateIndex])
    
    old_value = qTable[prevState][actionNumber]
    print(old_value)
    new_value=(1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    
    qTable[prevState][actionNumber] = new_value

    prevState=stateIndex

    if np.random.uniform(0, 1) < epsilon:
            actionNumber = random.randint(0,len(actions))# Explore action space
    else:
            actionNumber= np.argmax(qTable[stateIndex]) # Exploit learned values

    print(numbertoAction[actionNumber])
        
    feedback=input("Helpful Y/N? or Quit Q").lower()
    if feedback=='y':
        reward=1
    elif feedback=='n':
        reward=-1
    else:
        episodeRun=False
    print(new_value)

#print(qTable)
#print(qTable)


# In[9]:


#plot the map to see differences
import matplotlib.pyplot as plt
plt.imshow( qTable, cmap = 'rainbow' , interpolation = 'bilinear')
plt.title("Matplotlib PLot NumPy Array") 


# In[10]:


with open('test.npy', 'wb') as f:
    np.save(f, qTable)

