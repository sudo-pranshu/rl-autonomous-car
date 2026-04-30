import pygame
import random
import numpy as np

WIDTH, HEIGHT = 800, 600
LANE_LEFT, LANE_RIGHT = 200, 600
LANE_CENTER = (LANE_LEFT + LANE_RIGHT) / 2

class CarEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.car_x = LANE_CENTER
        self.car_y = 500
        self.velocity = 5
        self.obstacles = []
        self.spawn_timer = 0
        return self.get_state()

    def spawn_obstacle(self):
        self.obstacles.append([random.choice([250, 350, 450, 550]), -60])

    def step(self, action):
        reward = 0
        done = False

        # ===== DIRECT CONTROL (FIX) =====
        if action == 0:   # left
            self.car_x -= 6
        elif action == 1: # right
            self.car_x += 6
        elif action == 2: # accelerate
            self.velocity += 0.2
        elif action == 3: # brake
            self.velocity -= 0.2

        self.velocity = max(3, min(self.velocity, 7))

        # ===== WORLD =====
        self.spawn_timer += 1
        if self.spawn_timer > 25:
            self.spawn_obstacle()
            self.spawn_timer = 0

        for obs in self.obstacles:
            obs[1] += self.velocity

        self.obstacles = [o for o in self.obstacles if o[1] < HEIGHT + 50]

        # ===== COLLISION =====
        car_rect = pygame.Rect(self.car_x, self.car_y, 40, 60)
        for obs in self.obstacles:
            if car_rect.colliderect(pygame.Rect(obs[0], obs[1], 40, 60)):
                return self.get_state(), -200, True

        # ===== HARD BOUNDARY =====
        if self.car_x < LANE_LEFT:
            self.car_x = LANE_LEFT
            reward -= 50

        if self.car_x > LANE_RIGHT:
            self.car_x = LANE_RIGHT
            reward -= 50

        # ===== STRONG CENTERING =====
        dist = abs(self.car_x - LANE_CENTER)
        reward += 10 - (dist * 0.1)

        # ===== FORWARD =====
        reward += 3

        return self.get_state(), reward, False

    def cast_rays(self):
        distances = []
        for angle in [-60, -30, 0, 30, 60]:
            dist = 0
            for i in range(1, 60):
                x = self.car_x
                y = self.car_y - i * 5

                for obs in self.obstacles:
                    if pygame.Rect(obs[0], obs[1], 40, 60).collidepoint(x, y):
                        break

                dist = i
            distances.append(dist)

        return distances

    def get_state(self):
        return np.array(self.cast_rays() + [self.velocity], dtype=np.float32)

    def draw_rays(self, screen):
        for i in range(1, 60):
            pygame.draw.circle(screen, (0,200,255),
                (int(self.car_x+20), int(self.car_y - i*5)), 2)
