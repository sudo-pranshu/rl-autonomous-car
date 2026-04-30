import math

def update_position(x, y, angle, velocity):
    x += math.sin(math.radians(angle)) * velocity
    y -= math.cos(math.radians(angle)) * velocity
    return x, y

def apply_friction(velocity):
    return max(2, velocity * 0.98)
