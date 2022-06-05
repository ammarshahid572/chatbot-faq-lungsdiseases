diseases=['covid' , 'tuberculosis', 'asthma']
context1=['diagnosis', 'medications','types', 'facts']
context2=['sideeffects', 'preparation', 'duration']


diseasesKeywords= {
    "covid": ["covid", "covid-19", "covid19", "sars-cov"],
    "tuberculosis": ['tuberculosis', 'tb'],
    "asthma" : ["asthma"],
    "nocontext": [""]
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

def printContext(text):
    diseaseaseContext="nocontext"
    contexta="nocontext"
    contextb="nocontext"
    for diseasease in diseasesKeywords:
        for keyword in diseaseKeywords[diseasease]:
            if keyword in text:
                diseaseaseContext=diseasease
                break
    for context in context1:
        for keyword in context:
            if keyword in text:
                contexta=context
                break
    for context in context2:
        for keyword in context:
            if keyword in text:
                contextb=context
                break
    return(diseaseaseContext, contexta, contextb)

text="side effects"
print(printContext(text))
