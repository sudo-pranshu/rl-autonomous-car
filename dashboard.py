import pygame

def draw_dashboard(screen, reward, epsilon, velocity):
    font = pygame.font.SysFont("Arial", 18)

    texts = [
        f"Reward: {reward:.2f}",
        f"Epsilon: {epsilon:.2f}",
        f"Speed: {velocity:.2f}"
    ]

    for i, t in enumerate(texts):
        screen.blit(font.render(t, True, (255,255,255)), (10, 10 + i*20))


