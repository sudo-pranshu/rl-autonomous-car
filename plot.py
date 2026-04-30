import matplotlib.pyplot as plt

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title("Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("training.png")
