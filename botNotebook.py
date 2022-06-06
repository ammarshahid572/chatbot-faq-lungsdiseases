#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
    "asthmaMeds":["asthma meds"]

}


# In[4]:


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


# In[5]:


states= list()
for key in sorted(mainQuery):
    for context in sorted(context1):
        for  intent in sorted(context2):
            states.append((key,context, intent))
print(states[7])


# In[6]:


print (len(states))
print (len(actions))



# In[7]:


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



# In[8]:


#reward map----- old state to new state reward
reward= np.zeros((len(states), len(states)),dtype=float)
print(reward.shape)
# since we know we have to somehow end the convo. lets set maximum reward if the next state is ending the convo
for i,lastState in enumerate(states):
    for j,nextState in enumerate(states):
        if "endNote" in nextState:
            reward[i][j] =1


# In[9]:


#start q learn algo
# Initialize parameters
gamma = 0.75 # Discount factor 
alpha = 0.9 # Learning rate 


# In[10]:


qTable= np.zeros((len(states), len(actions)), dtype=float)


# In[13]:


#lets defie an episode
episodeRun=True
while episodeRun:
    text=input("")
    state=getContext(text)
    stateIndex=statetoNumber[state]
    actionNumber=np.argmax(qTable[stateIndex])
    print(actions[numbertoAction[actionNumber]])

    feedback=input("Helpful Y/N? or Quit Q").lower()
    if feedback=='y':
        qTable[stateIndex][actionNumber]+=1
    elif feedback=='n':
        qTable[stateIndex][actionNumber]-=1
    else:
        episodeRun=False

print(qTable)


# In[14]:


#plot the map to see differences
import matplotlib.pyplot as plt
plt.imshow( qTable, cmap = 'rainbow' , interpolation = 'bilinear')
plt.title("Matplotlib PLot NumPy Array") 

