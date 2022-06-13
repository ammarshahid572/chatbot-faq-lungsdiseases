

from enum import Flag
import random
import numpy as np


class Chatbot:
    prevContext= ("nocontext", "nocontext", "nocontext", "nocontext", "nocontext")
    
    mainQuery= {
        "greeting": ["Hi", "Hello", "How are you?"],
        "covid": ["covid", "covid-19", "covid19", "sars-cov"],
        "tuberculosis": ['tuberculosis', 'tb'],
        "asthma" : ["asthma"],
        "nocontext": [""],
        "endNote": ["Thank you", "This was helpful!", "See you later!"]
    }
    context1 = {
        "diagnosis":["if i have", "if he has", "if she has", "if they have", "diagnosis", "diagnosed"],
        "medications": ["treatment", "medications", "medicine"],
        "symptoms":["symptoms"],
        "types":["types", "variants", "stages"],
        "nocontext":[""]
    }

    context2= {
        "sideeffects":["side effects", "careful", "effects"],
        "duration": ["how long", "duration"],
        "nocontext":[""]
    }

    actions= {
        "greeting": ["Hello"],
        "whatCanIhelp":["Sure, what can I help you with"],
        "asthmaDef": ["Asthma is a disease that affects your lungs. It is one of the most common long-term diseases of children, but adults can have asthma, too. Asthma causes wheezing, breathlessness, chest tightness, and coughing at night or early in the morning. "],
        "covidDef": ["Coronaviruses are a family of viruses that can cause respiratory illness in humans. They are called “corona” because of crown-like spikes on the surface of the virus."],
        "tbDef": ["Tuberculosis (TB) is a bacterial infection spread through inhaling tiny droplets from the coughs or sneezes of an infected person. It mainly affects the lungs, but it can affect any part of the body, including the tummy (abdomen), glands, bones and nervous system."],
        "endNote": ["Thanks for chatting"],
        "covidDiag":["Covid is most accurately detected using PCR tests."],
        "asthmaDiag":["There are 2 main tests to diagnose Asthma: FeNo and Spirometry"],
        "tbDiag":["There are two basic tests used for TB. TB Skin Test and TB Blood Test."],
        "noContext": ["Sorry didnt understand the question. please specify in details or try rephrasing."],
        "covidvariants": ["Sars-Cov-2 has had many circulating virus namely Alpha, Beta, Gamma, Delta and Omicron"],
        "ashtmavariants": ["Asthma has these common types: Allergic Asthma, Non-allergic Asthma, Cough-variant asthma, Noturnal and Occupational Asthma"],
        "tbvariants":["There are two types of TB conditions: TB disease and latent TB infection. "],
        "covidMeds": ["Scientists around the world are working to find and develop treatments for COVID-19. However preventive vaccinations are commonly available."],
        "tbMeds": ["The most common treatment for active TB is isoniazid INH in combination with three other drugs—rifampin, pyrazinamide and ethambutol."],
        "asthmaMeds":["There's currently no cure for asthma, but treatment can help control the symptoms so you're able to live a normal, active life. Inhalers, which are devices that let you breathe in medicine, are the main treatment. Tablets and other treatments may also be needed if your asthma is severe."],
        "asthmaduration": ["Asthma symptoms can show up for a few times a month to several times a day/"],
        "covidDuration":["Generally covid lasts for about 2 weeks."],
        "tbduration": ["TB treatment can take anywhere between 6 to 9 months"],
        "asthmaMedSide": ["The most common side effects of inhaled preventer medication (inhaled corticosteroids) are a hoarse voice, sore mouth and throat, and fungal infections of the throat. "],
        "covidMedSide":["Covid Vaccination has minor side effects ranging for mild fever and weakness"],
        "tbMedSide":["some side effects of TB treatments are itchy skin ,skin rashes, bruising or yellow skin, upset stomach, nausea, vomiting, diarrhoea or loss of appetite"],
        "asthmaSymptoms":["Asthma attacks signs include wheezing, coughing and chest tightness becoming severe and constant, being too breathless to eat, speak or sleep, drowsiness, confusion, exhaustion or dizziness"],
        "covidSymptoms":["Most common symptoms of covid are fever, cough, tiredness,loss of taste or smell"],
        "tbSymptoms":["Some symptoms of TB are a persistent cough that lasts more than 3 weeks and usually brings up phlegm, which may be bloody, weight loss, night sweats, high temperature, tiredness and fatigue."],
        "unrelated": ["sorry I can only answer for question I am allowed to."],
    }

    gamma = 0.75 # Discount factor 
    alpha = 0.9 # Learning rate 
    epsilon = 0.1
    next_stateIndex=0
    stateIndex=0
    actionNumber=0
    episodeRun=True

    #-----------setup----------------------------------------------
    states= list()
    for key in sorted(mainQuery):
        for context in sorted(context1):
            for  intent1 in sorted(context2):
                for intents2 in sorted(context1):
                    for intent3 in sorted(context2):
                        states.append((key, context, intent1, intents2, intent3))
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
    
    prevState=statetoNumber[prevContext]
    stateIndex=prevState
    old_action=0
    new_action=0
    old_value=0
    next_max=0
    reward_value=0
    qTable= np.zeros((len(states), len(actions)), dtype=float)
    
    #-------methods-------------------------------------

    def update(self):
        self.old_action=self.new_action
        self.prevState=self.stateIndex
        pass
    def reward(self,value):
        self.reward_value=value
        pass
    
    def getAction(self, text):
        context= self.getContext(text)
        self.stateIndex=self.statetoNumber[context]

        self.old_value=self.qTable[self.prevState][self.old_action]
        
        self.next_max = np.max(self.qTable[self.stateIndex])
        new_value=(1 - self.alpha) * self.old_value + self.alpha * (self.reward_value + self.gamma * self.next_max)
        self.qTable[self.prevState][self.old_action]=new_value


        if np.random.uniform(0, 1) < self.epsilon:
            self.new_action= random.randint(0,len(self.actions))# Explore action space
        else:
            self.new_action= np.argmax(self.qTable[self.stateIndex]) # Exploit learned values
        
        return self.actions[self.numbertoAction[self.new_action]]
    
    def saveQtable(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.qTable)
    

    def loadQtable(self, filename):
        try:
            with open(filename,'rb')as f:
                f.seek(0)
                loadedFile=np.load(f ,allow_pickle=False)
                if loadedFile.shape== self.qTable.shape:
                    self.qTable=loadedFile
                    print("Loaded old backup")
                else:
                    print("File invalid")
        except FileNotFoundError:
            print("File Not found")
    
    def getContext(self,text):
        text=text.lower()
        keys= list(self.prevContext)
        for key in self.mainQuery:
            for keyword in self.mainQuery[key]:
                if keyword.lower() in text:
                    if key=="nocontext":
                        continue
                    keys[0]=key
                    break
            else:
                continue
            break
        for key in self.context1:
            for keyword in self.context1[key]:
                if keyword.lower()  in text:
                    if key=="nocontext":
                        continue
                    keys[3]=key
                    break
            else:
                continue
            break
        for key in self.context2:
            for keyword in self.context2[key]:
                if keyword.lower()  in text:
                    if key=="nocontext":
                        continue
                    keys[4]=key
                    break
            else:
                continue
            break
        return (keys[0], keys[1], keys[2], keys[3], keys[4])

test=False
if (test==True):
    bot = Chatbot()
    bot.loadQtable('test.npy')
    while True:
        text=input("Enter text:").lower()
        action=bot.getAction(text)
        print(action)

        k=input("Helpful Y/N or Quit Q").lower()
        if k=='y':
            bot.reward(1)
        elif k=='n':
            bot.reward(-1)
        elif k=='q':
            bot.reward(0)
            break
        else:
            bot.reward(0)
            pass
        bot.update()
    bot.saveQtable('test.npy')
