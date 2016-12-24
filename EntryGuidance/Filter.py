''' Defines filters for estimation '''



def FadingMemory(currentValue, measuredValue, gain):
    return (1-gain)*(measuredValue-currentValue)
    
