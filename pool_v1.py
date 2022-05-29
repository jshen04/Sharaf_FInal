import pymunk
import pygame
import pymunk.pygame_util
import math
from pygame.locals import USEREVENT, QUIT, KEYDOWN, KEYUP, K_s, K_r, K_q, K_ESCAPE, K_UP, K_DOWN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import itertools

BALL_MASS = 1
BALL_RADIUS = 25
BALL_FRICTION = 0
BALL_ELASTICITY = 1

WORLD_DIMS = (1200, 600)
GATE_ELASTICITY = 1
GATE_FRICTION = 0

FRICTION_FORCE = 1

class Ball:
    def __init__(self, position, collision_type=1):
        ball_body = pymunk.Body(mass=BALL_MASS, moment=math.inf)
        # ball_body = pymunk.Body(mass=BALL_MASS, moment=pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
        ball_body.position = position
        ball_shape = pymunk.Circle(ball_body, radius=BALL_RADIUS)
        ball_shape.friction = BALL_FRICTION
        ball_shape.elasticity = BALL_ELASTICITY
        ball_shape.collision_type = collision_type
        self.velocity = pymunk.Vec2d(0, 0)
        self.shape = ball_shape
        self.shape.color = (169, 0, 0, 255)

    def set_position(self, position):
        self.shape.body.position = self.set_position(position)

class CueBall(Ball):
    def __init__(self, position, collision_type=2):
        super().__init__(position, collision_type=collision_type)
        self.shape.color = (255, 255, 255, 255)

def initialize_border(space):
    t = pymunk.Body(body_type=pymunk.Body.STATIC)
    t.elasticity = GATE_ELASTICITY
    t.friction = GATE_FRICTION
    t.position = (WORLD_DIMS[0]/2, WORLD_DIMS[1])
    ts = pymunk.Poly.create_box(body=t, size=(WORLD_DIMS[0], BALL_RADIUS))
    ts.collision_type = 3
    ts.color = (75, 55, 28, 255)

    b = pymunk.Body(body_type=pymunk.Body.STATIC)
    b.elasticity = GATE_ELASTICITY
    b.friction = GATE_FRICTION
    b.position = (WORLD_DIMS[0]/2, 0)
    bs = pymunk.Poly.create_box(body=b, size=(WORLD_DIMS[0], BALL_RADIUS))
    bs.collision_type = 3
    bs.color = (75, 55, 28, 255)

    l = pymunk.Body(body_type=pymunk.Body.STATIC)
    l.elasticity = GATE_ELASTICITY
    l.friction = GATE_FRICTION
    l.position = (0, WORLD_DIMS[1]/2)
    ls = pymunk.Poly.create_box(body=l, size=(BALL_RADIUS, WORLD_DIMS[1]))
    ls.collision_type = 4
    ls.color = (75, 55, 28, 255)

    r = pymunk.Body(body_type=pymunk.Body.STATIC)
    r.elasticity = GATE_ELASTICITY
    r.friction = GATE_FRICTION
    r.position = (WORLD_DIMS[0], WORLD_DIMS[1]/2)
    rs = pymunk.Poly.create_box(body=r, size=(BALL_RADIUS, WORLD_DIMS[1]))
    rs.collision_type = 4
    rs.color = (75, 55, 28, 255)

    space.add(ts, bs, ls, rs)
    space.add(t, b, l, r)


def initialize_holes(space):
    for i in range(6):
        hole = bot_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        hole.position = (WORLD_DIMS[0] * ((i % 3)/2), WORLD_DIMS[1] * (i//3))
        hole_shape = pymunk.Circle(bot_body, radius=BALL_RADIUS * 1.5)
        hole_shape.collision_type = 5
        hole_shape.color = (0, 0, 0, 255)
        space.add(hole, hole_shape)


def detect_collisions(b1, b2):
    return np.linalg.norm(b1.shape.body.position - b2.shape.body.position) <= 2 * BALL_RADIUS

def elastic_collision(b1, b2):
    m1 = b1.shape.body.mass
    m2 = b2.shape.body.mass
    v1x, v1y = b1.velocity
    v2x, v2y = b2.velocity

    c1 = ((2 * m2) / (m1 + m2)) * np.dot((b1.velocity[0] - b2.velocity[0], b1.velocity[1] - b2.velocity[1]), (b1.shape.body.position - b2.shape.body.position)) / (np.linalg.norm(b1.shape.body.position - b2.shape.body.position) ** 2)
    v1f = (v1x - c1 * (b1.shape.body.position[0] - b2.shape.body.position[0]), v1y - c1 * (b1.shape.body.position[1] - b2.shape.body.position[1]))

    c2 = ((2 * m1) / (m1 + m2)) * np.dot((b2.velocity[0] - b1.velocity[0], b2.velocity[1] - b1.velocity[1]), (b2.shape.body.position - b1.shape.body.position)) / (np.linalg.norm(b2.shape.body.position - b1.shape.body.position) ** 2)
    v2f = (v2x - c2 * (b2.shape.body.position[0] - b1.shape.body.position[0]), v2y - c2 * (b2.shape.body.position[1] - b1.shape.body.position[1]))

    b1.velocity = v1f
    b2.velocity = v2f

    # v1xf = ((m1 - m2) / (m1 + m2)) * v1x + ((2 * m2) / (m1 + m2)) * v2x
    # v2xf = ((2 * m1) / (m1 + m2)) * v1x - ((m1 - m2) / (m1 + m2)) * v2x
    #
    # v1yf = ((m1 - m2) / (m1 + m2)) * v1y + ((2 * m2) / (m1 + m2)) * v2y
    # v2yf = ((2 * m1) / (m1 + m2)) * v1y - ((m1 - m2) / (m1 + m2)) * v2y
    #
    # b1.velocity = pymunk.Vec2d(v1xf, v1yf)
    # b2.velocity = pymunk.Vec2d(v2xf, v2yf)
    return True

def friction(b, friction_accel_magnitude, dt):
    v = math.sqrt(b.velocity[0]**2 + b.velocity[1]**2)
    if v != 0:
        b.velocity = (b.velocity[0] - friction_accel_magnitude * dt * b.velocity[0] / v, b.velocity[1] - friction_accel_magnitude * dt * b.velocity[1] / v)
    if abs(v) <= 0.01:
        b.velocity = (0, 0)

def main():
    dt = 0.01
    pygame.init()
    screen = pygame.display.set_mode(WORLD_DIMS)
    screen.fill((0, 102, 0))
    clock = pygame.time.Clock()
    running = True
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    space = pymunk.Space(WORLD_DIMS)
    space.gravity = 0, 0
    initialize_border(space)
    initialize_holes(space)

    balls = []

    cueball = CueBall(position=(1000, 300))
    space.add(cueball.shape, cueball.shape.body)
    # cueball.shape.body.velocity = pymunk.Vec2d(-100, 0)
    cueball.velocity = pymunk.Vec2d(-100, 0)
    balls.append(cueball)

    for i in range(1, 6, 1):
        for j in range(1, i + 1, 1):
            x = (3 - i) * BALL_RADIUS * 1.05 * math.sqrt(3) + 300
            y = (j - 3) * BALL_RADIUS * 2 * 1.05 + 300 + ((5 - i) * BALL_RADIUS * 1.05)
            ball = Ball(position=(x, y))
            space.add(ball.shape, ball.shape.body)
            balls.append(ball)

    kes = []
    ke = 0
    xs = []
    ys = []
    for ball in balls:
        ke = ke + .5 * ball.shape.body.mass * (np.linalg.norm(ball.velocity) ** 2)
    while running:
        for event in pygame.event.get():
            if len(balls) == 1:
                running = False
            elif ke == 0:
                running = False
            elif event.type == QUIT or (event.type == KEYDOWN and event.key in (K_q, K_ESCAPE)):
                running = False
            elif event.type == KEYDOWN and event.key == K_s:
                # Start/stop simulation.
                running = not running

        contact_pairs = []
        for pair in itertools.combinations(balls, 2):
            if detect_collisions(pair[0], pair[1]):
                contact_pairs.append(pair)

        for pair in contact_pairs:
            elastic_collision(pair[0], pair[1])

        for ball in balls:
            x = ball.shape.body.position[0]
            y = ball.shape.body.position[1]
            if ((x)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
            elif ((x-600)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
            elif ((x-1200)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
            elif ((x)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
            elif ((x-600)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
            elif ((x-1200)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
            else:
                if x <= 1.5 * BALL_RADIUS:
                    ball.velocity = (-1 * ball.velocity[0], ball.velocity[1])
                if x >= 1200 - 1.5 * BALL_RADIUS:
                    ball.velocity = (-1 * ball.velocity[0], ball.velocity[1])
                if y <= 600 - 1.5 * BALL_RADIUS:
                    ball.velocity = (ball.velocity[0], -1 * ball.velocity[1])
                if y >= 1.5 * BALL_RADIUS:
                    ball.velocity = (ball.velocity[0], -1 * ball.velocity[1])
                friction(ball, FRICTION_FORCE/ball.shape.body.mass, dt)

        screen.fill((0, 102, 0))
        space.debug_draw(draw_options)
        pygame.display.update()
        space.step(dt)
        ke = 0
        x = 0
        y = 0
        n = len(balls)
        for ball in balls:
            ball.shape.body.position = (ball.shape.body.position[0] + ball.velocity[0] * dt, ball.shape.body.position[1] + ball.velocity[1] * dt)
            ke = ke + .5 * ball.shape.body.mass * (np.linalg.norm(ball.velocity) ** 2)
            x = x + ball.shape.body.position[0]
            y = y + ball.shape.body.position[1]
        xs.append(x/n)
        ys.append(y/n)
        kes.append(ke)

    plt.plot(kes)
    plt.show()

    plt.plot(xs, ys)
    plt.show()

if __name__ == '__main__':
    main()
