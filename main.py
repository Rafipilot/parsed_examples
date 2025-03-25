import ao_pyth as ao
from config import API_KEY

# Initialize architecture with predefined 5 input neurons, 5 q(hidden) neurons, 5 output neurons. 
#The 5 output neurons correspond to the likelihood of buying (scale 1-5)
arch = ao.Arch(arch_i="[1, 1, 1, 1, 1]", arch_z="[1, 1, 1, 1, 1]", api_key=API_KEY, kennel_id="Parsed_DEMO2") #Create new arch  in our api, if not already created.
print(arch.api_status)
# Create an agent with the given architecture
agent = ao.Agent(arch, uid="2")

# Training examples: Each input set follows the format:
# [Payment setup, Item in basket, User logged in, User new, User returning]
# Corresponding label: Likelihood of buying (scale 1-5)


###UNCOMMENT TO TRAIN THE AGENT###
# training_data = [
#     ([1, 1, 1, 1, 0], [1,1,1,1,0]),  # New user, logged in, payment setup, has item – high chance
#     ([1, 1, 1, 0, 1], [1,1,1,1,1]),  # Returning user, logged in, payment setup, has item – very high chance
#     ([1, 1, 0, 1, 0], [1,1,1,0,0]),  # New user, item in basket, but not logged in – medium chance
#     ([1, 0, 1, 0, 1], [1,1,0,0,0]),  # Returning user, logged in, no item in basket – low chance
#     ([0, 1, 1, 0, 1], [1,1,1,1,0]),  # Returning user, logged in, item in basket, no payment setup – good chance
#     ([1, 1, 0, 0, 1], [1,1,1,0,0]),  # Returning user, item in basket, but not logged in – medium chance
#     ([0, 0, 1, 0, 1], [1,0,0,0,0]),  # Returning user, logged in, no item, no payment – very low chance
#     ([1, 1, 1, 0, 0], [1,1,1,1,0]),  # Not new or returning, logged in, item in basket – good chance
#     ([0, 1, 0, 1, 0], [1,1,0,0,0]),  # New user, item in basket, no login, no payment – low chance
#     ([1, 0, 0, 0, 1], [1,0,0,0,0]),  # Returning user, no item, not logged in – very low chance
# ]

# # Train the agent with the examples
# for inp, label in training_data:
#     agent.next_state(INPUT=inp, LABEL=label)

# Example inference
response = agent.next_state([1, 1, 1, 0, 1])  # Predict likelihood for a user with payment setup, item in basket, logged in, returning user


##Get number of ones in response to calculate percentage.
ones = 0
for item in enumerate(response):
    if item[1] == 1:
        ones +=1

print("Predicted likelihood of buying: ", ones / len(response)*100, "%")