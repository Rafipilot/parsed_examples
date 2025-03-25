import ao_pyth as ao
from config import API_KEY

# Initialize architecture with predefined 4 input neurons, 5 hidden neurons, 5 output neurons. 
# The 5 output neurons correspond to the likelihood of buying (scale 1-5)
arch = ao.Arch(arch_i="[1, 1, 1, 1]", arch_z="[1, 1, 1, 1, 1]", api_key=API_KEY, kennel_id="Parsed_DEMO3") 
print(arch.api_status)

# Create an agent with the given architecture
agent = ao.Agent(arch, uid="test1")

# Training examples: 
# Format: [Payment setup, Item in basket, User logged in, User new] -> Likelihood of buying (scale 1-5)
training_data = [
    ([1, 1, 1, 0], [1, 1, 1, 1, 1]),  # High likelihood (returning user, logged in, has item)
    ([1, 1, 0, 0], [1, 1, 1, 0, 1]),  # Medium likelihood (not logged in)
    ([0, 1, 1, 0], [1, 1, 0, 1, 0]),  # Medium-low likelihood (no payment setup)
    ([1, 0, 1, 1], [1, 0, 1, 0, 0]),  # Low likelihood (no item in basket)
    ([0, 0, 0, 1], [0, 0, 1, 0, 0]),  # Very low likelihood (new user, no setup, no item)
    ([1, 1, 1, 1], [1, 1, 1, 1, 1]),  # Highest likelihood (everything is optimal)
    ([0, 1, 1, 1], [1, 1, 0, 1, 0]),  # Medium-low likelihood (new user)
    ([1, 0, 0, 0], [0, 1, 0, 0, 0]),  # Very low likelihood (no basket, not logged in)
    ([1, 1, 0, 1], [1, 0, 1, 0, 1]),  # Medium likelihood (new user but has setup)
    ([1, 0, 1, 0], [0, 1, 0, 1, 0])   # Low likelihood (no item)
]

# Train the agent with the examples
###Uncomment to train the agent
# for inp, label in training_data:
#     agent.next_state(INPUT=inp, LABEL=label, unsequenced=True)

# Example inference
response = agent.next_state([0, 1, 1, 0], unsequenced=True)  # Predict likelihood for a user with payment setup, item in basket, logged in, returning user

# Calculate percentage of ones in response
ones = sum(1 for item in response if item == 1)

print("Predicted likelihood of buying: ", ones / len(response) * 100, "%")
