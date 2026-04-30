# ================= RL DEFINITIONS =================
import pygame
import random
import matplotlib.pyplot as plt
import numpy as np

pygame.init()

WIDTH, HEIGHT = 800, 600
LANES = [250, 350, 450, 550]

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

car_lane_index = 1
car_x = LANES[car_lane_index]
car_y = 500

velocity = 5
target_velocity = 5
max_speed = 8
min_speed = 3

mode = "AI"

obstacles = []
spawn_timer = 0

target_lane_index = car_lane_index
switching = False
lane_lock = 0

distance_travelled = 0

speed_history = []
distance_history = []
time_steps = []
t = 0

Q = np.zeros((8, 4))
alpha = 0.1
gamma = 0.9
epsilon = 1.0

episode = 1

def spawn_obstacle():
    obstacles.append([random.choice(LANES), -80])

def get_state():
    obstacle = 0
    for obs in obstacles:
        if 0 < car_y - obs[1] < 150 and abs(obs[0] - car_x) < 50:
            obstacle = 1
    return car_lane_index * 2 + obstacle

running = True

while running:
    screen.fill((20,20,25))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_m:
                mode = "MANUAL" if mode == "AI" else "AI"

    spawn_timer += 1
    if spawn_timer > 32:
        spawn_obstacle()
        spawn_timer = 0

    for obs in obstacles:
        obs[1] += velocity

    state = get_state()

    if random.random() < epsilon:
        action = random.randint(0,3)
    else:
        action = np.argmax(Q[state])

    danger = False
    closest_dist = 999

    for obs in obstacles:
        dist = car_y - obs[1]
        future_dist = dist - velocity * 6

        if 0 < future_dist < 350 and abs(obs[0] - car_x) < 60:
            danger = True
            closest_dist = min(closest_dist, future_dist)

    if danger:
        if closest_dist < 80:
            target_velocity = 3
        elif closest_dist < 160:
            target_velocity = 4
        elif closest_dist < 260:
            target_velocity = 5
    else:
        target_velocity = max_speed

    velocity += (target_velocity - velocity) * 0.05
    velocity = max(min_speed, min(max_speed, velocity))

    if not switching and danger:
        safe_lanes = []

        for i, lane in enumerate(LANES):
            safe = True

            for obs in obstacles:
                dist = car_y - obs[1]
                future_dist = dist - velocity * 6

                if 0 < future_dist < 300 and abs(obs[0] - lane) < 65:
                    safe = False
                    break

            if safe:
                safe_lanes.append(i)

        if safe_lanes:
            target_lane_index = min(
                safe_lanes, key=lambda i: abs(i - car_lane_index)
            )
            if target_lane_index != car_lane_index:
                switching = True
                lane_lock = 20

    if not danger and not switching and lane_lock == 0:
        target_lane_index = car_lane_index

    for obs in obstacles:
        future_dist = (car_y - obs[1]) - velocity * 5
        if 0 < future_dist < 200 and abs(obs[0] - LANES[target_lane_index]) < 65:
            target_lane_index = car_lane_index
            switching = False
            break

    rl_weight = min(0.25, t / 5000)

    if not danger and closest_dist > 120 and lane_lock == 0:
        if action == 0 and car_lane_index > 0 and random.random() < rl_weight:
            target_lane_index = car_lane_index - 1
            switching = True
            lane_lock = 20

        elif action == 1 and car_lane_index < 3 and random.random() < rl_weight:
            target_lane_index = car_lane_index + 1
            switching = True
            lane_lock = 20

    if action == 2:
        target_velocity = min(max_speed, target_velocity + 0.04 * rl_weight)

    elif action == 3:
        target_velocity = max(min_speed, target_velocity - 0.04 * rl_weight)

    target_x = LANES[target_lane_index]
    next_x = car_x + (target_x - car_x) * 0.12

    blocked = False
    for obs in obstacles:
        future_y = obs[1] + velocity * 5

        if abs(future_y - car_y) < 130 and abs(obs[0] - next_x) < 65:
            blocked = True
            target_velocity = min(target_velocity, 3)
            break

    if not blocked:
        car_x = next_x

    if abs(car_x - target_x) < 2:
        car_lane_index = target_lane_index
        switching = False

    if lane_lock > 0:
        lane_lock -= 1

    car_rect = pygame.Rect(car_x, car_y, 40, 70)
    collision = False

    for obs in obstacles:
        obs_rect = pygame.Rect(obs[0], obs[1], 40, 70)
        if car_rect.colliderect(obs_rect):
            collision = True

    reward = 1
    for obs in obstacles:
        if abs(obs[1] - car_y) < 100 and abs(obs[0] - car_x) < 60:
            reward -= 5

    if collision:
        reward -= 100
        episode += 1

        car_lane_index = 1
        car_x = LANES[car_lane_index]
        target_lane_index = car_lane_index
        switching = False
        lane_lock = 0

        velocity = 5
        target_velocity = 5

        obstacles.clear()
        spawn_timer = 0

    next_state = get_state()

    Q[state][action] += alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state][action]
    )

    epsilon = max(0.05, epsilon * 0.998)

    distance_travelled += velocity * 0.1

    # ===== DRAW =====
    for i in range(HEIGHT):
        c = 20 + int(i * 0.03)
        pygame.draw.line(screen, (c, c, c+5), (0, i), (WIDTH, i))

    pygame.draw.rect(screen, (45,45,45), (200,0,400,600))

    offset = pygame.time.get_ticks() // 50 % 40
    for lane_x in LANES:
        for y in range(-40, 600, 40):
            pygame.draw.line(screen, (200,200,200),
                (lane_x, y+offset), (lane_x, y+20+offset), 2)

    # ===== SENSOR + DISTANCE + DANGER ZONE =====
    font_small = pygame.font.SysFont("Arial", 14)

    for obs in obstacles:
        dist = car_y - obs[1]

        if 0 < dist < 300:
            pygame.draw.line(screen, (0,200,255),
                (car_x+20, car_y),
                (obs[0]+20, obs[1]), 1)

            label = font_small.render(str(int(dist)), True, (0,200,255))
            screen.blit(label, (obs[0]+10, obs[1]-10))

        # collision zone highlight
        if abs(obs[1] - car_y) < 120 and abs(obs[0] - car_x) < 65:
            pygame.draw.rect(screen, (255,100,100),
                (obs[0], obs[1], 40, 70), 2)

    brake = target_velocity < velocity

    pygame.draw.rect(screen,(0,255,150),(car_x,car_y,40,70),border_radius=10)
    pygame.draw.rect(screen,(0,200,120),(car_x+5,car_y+8,30,20),border_radius=5)

    pygame.draw.circle(screen,(255,255,200),(int(car_x+8),int(car_y)),3)
    pygame.draw.circle(screen,(255,255,200),(int(car_x+32),int(car_y)),3)

    if brake:
        intensity = min(255, int((velocity - target_velocity) * 200))
        color = (255, intensity, intensity)
        pygame.draw.circle(screen,color,(int(car_x+8),int(car_y+70)),4)
        pygame.draw.circle(screen,color,(int(car_x+32),int(car_y+70)),4)

    for obs in obstacles:
        pygame.draw.rect(screen,(200,0,0),(obs[0],obs[1],40,70),border_radius=10)
        pygame.draw.rect(screen,(150,0,0),(obs[0]+5,obs[1]+8,30,20),border_radius=5)

    font = pygame.font.SysFont("Arial", 18)
    screen.blit(font.render(f"Episode: {episode}", True, (255,255,255)), (10,10))
    screen.blit(font.render(f"Speed: {round(velocity,1)}", True, (255,255,255)), (10,30))
    screen.blit(font.render(f"Distance: {int(distance_travelled)}", True, (255,255,255)), (10,50))
    screen.blit(font.render(f"Q: {np.round(Q[state],2)}", True, (255,255,0)), (10,70))

    pygame.display.flip()
    clock.tick(30)

    t += 1
    speed_history.append(velocity)
    distance_history.append(distance_travelled)
    time_steps.append(t)

pygame.quit()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(time_steps, speed_history)
plt.subplot(1,2,2)
plt.plot(time_steps, distance_history)
plt.tight_layout()
plt.show()

print("Final Q-table:")
print(Q)

