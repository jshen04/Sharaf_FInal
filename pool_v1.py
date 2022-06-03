import pymunk
import pygame
import pymunk.pygame_util
import math
from pygame.locals import USEREVENT, QUIT, KEYDOWN, KEYUP, K_s, K_r, K_q, K_ESCAPE, K_UP, K_DOWN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import itertools
import random

BALL_MASS = 1
BALL_RADIUS = 25
BALL_FRICTION = 0
BALL_ELASTICITY = 1

WORLD_DIMS = (1200, 600)
GATE_ELASTICITY = 1
GATE_FRICTION = 0

FRICTION_FORCE = 1

class Ball:
    def __init__(self, position, design, collision_type=1):
        ball_body = pymunk.Body(mass=BALL_MASS, moment=math.inf)
        # ball_body = pymunk.Body(mass=BALL_MASS, moment=pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
        ball_body.position = position
        ball_shape = pymunk.Circle(ball_body, radius=BALL_RADIUS)
        ball_shape.friction = BALL_FRICTION
        ball_shape.elasticity = BALL_ELASTICITY
        ball_shape.collision_type = collision_type
        self.velocity = pymunk.Vec2d(0, 0)
        self.shape = ball_shape
        self.design = design # either solid (0), striped (1), or eight ball (8), or cue ball (-1)
        if self.design == 0:
            self.shape.color = (169, 0, 0, 255)
        if self.design == 1:
            self.shape.color = (0, 0, 169, 255)
        if self.design == 8:
            self.shape.color = (0, 0, 0, 255)
        if self.design == -1:
            self.shape.color = (255, 255, 255, 255)


    def set_position(self, position):
        self.shape.body.position = self.set_position(position)

class CueBall(Ball):
    def __init__(self, position, design=-1, collision_type=2):
        super().__init__(position, design=design, collision_type=collision_type)

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
    return True

def friction(b, friction_accel_magnitude, dt):
    v = math.sqrt(b.velocity[0]**2 + b.velocity[1]**2)
    if v != 0:
        b.velocity = (b.velocity[0] - friction_accel_magnitude * dt * b.velocity[0] / v, b.velocity[1] - friction_accel_magnitude * dt * b.velocity[1] / v)
    if abs(v) <= 0.01:
        b.velocity = (0, 0)

class player:
    def __init__(self, ix):
        self.ix = ix
        self.turn = False
        self.design = None
        self.olddesign = None


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
    balls.append(cueball)

    ix = 0
    ixs = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    solids = np.random.choice(ixs, 7, replace=False)
    for i in range(1, 6, 1):
        for j in range(1, i + 1, 1):
            if ix == 4:
                design = 8
            elif ix in solids:
                design = 0
            else:
                design = 1
            ix = ix + 1
            x = (3 - i) * BALL_RADIUS * 1.05 * math.sqrt(3) + 300
            y = (j - 3) * BALL_RADIUS * 2 * 1.05 + 300 + ((5 - i) * BALL_RADIUS * 1.05)
            ball = Ball(position=(x, y), design=design)
            space.add(ball.shape, ball.shape.body)
            balls.append(ball)

    # kes = []
    ke = 0
    # xs = []
    # ys = []
    new_iter = True
    for ball in balls:
        ke = ke + .5 * ball.shape.body.mass * (np.linalg.norm(ball.velocity) ** 2)

    solids = list(filter(None, [b if b.design == 0 else None for b in balls]))
    stripes = list(filter(None, [b if b.design == 1 else None for b in balls]))
    eightball = list(filter(None, [b if b.design == 8 else None for b in balls]))

    oldsolids = len(solids)
    oldstripes = len(stripes)
    oldeightball = len(eightball)

    p1 = player(1)
    p2 = player(2)
    p1.turn = True

    # indicates type of target ball (0 for solid 1 for stripe)
    player1 = -1
    player2 = -1

    # 1 for player 1, 2 for player 2
    togo = 1

    turn = 0
    trigger = False
    while running:
        for event in pygame.event.get():
            if len(balls) == 1:
                running = False
            elif ke == 0:
                new_iter = True
            elif event.type == QUIT or (event.type == KEYDOWN and event.key in (K_q, K_ESCAPE)):
                running = False
            elif event.type == KEYDOWN and event.key == K_s:
                # Start/stop simulation.
                running = not running

            if new_iter:
                if p1.turn:
                    print("Player 1 to move")
                if p2.turn:
                    print("Player 2 to move")
                new_pos = pymunk.pygame_util.get_mouse_pos(screen)
                diff = new_pos - cueball.shape.body.position
                dx, dy = diff[0], diff[1]
                if event.type == pygame.MOUSEBUTTONDOWN:
                    turn = turn + 1
                    oldeightball = len(eightball)
                    oldsolids = len(solids)
                    oldstripes = len(stripes)

                    if type(p1.design) == list:
                        p1.olddesign = len(p1.design)
                    if type(p2.design) == list:
                        p2.olddesign = len(p2.design)
                    if dx**2 + dy**2 >= 10000:
                        h = ((dx ** 2) + (dy ** 2)) ** 0.5
                        if not h == 0:
                            dx = 100 * dx / h
                            dy = 100 * dy / h
                            cueball.velocity = dx, dy
                            new_iter = False
                    else:
                        cueball.velocity = dx, dy
                        new_iter = False

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
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-600)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-1200)**2)+((y)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-600)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
            elif ((x-1200)**2)+((y-600)**2) <= (2.5*BALL_RADIUS)**2 and not type(ball) == CueBall:
                space.remove(ball.shape.body, ball.shape)
                balls.remove(ball)
                if ball in solids:
                    solids.remove(ball)
                if ball in stripes:
                    stripes.remove(ball)
                if ball in eightball:
                    eightball.remove(ball)
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

                # if type(p1.design) == list:
                #     if p1.olddesign - len(p1.design) > 0 and p1.turn:
                #         p1.turn = p1.turn
                #         p2.turn = p2.turn
                # if type(p2.design) == list:
                #     if p2.olddesign - len(p2.design) > 0 and p2.turn:
                #         p1.turn = p1.turn
                #         p2.turn = p2.turn

        screen.fill((0, 102, 0))
        space.debug_draw(draw_options)
        pygame.display.update()
        space.step(dt)
        ke = 0
        # x = 0
        # y = 0
        # n = len(balls)
        for ball in balls:
            ball.shape.body.position = (ball.shape.body.position[0] + ball.velocity[0] * dt, ball.shape.body.position[1] + ball.velocity[1] * dt)
            ke = ke + .5 * ball.shape.body.mass * (np.linalg.norm(ball.velocity) ** 2)
    #         x = x + ball.shape.body.position[0]
    #         y = y + ball.shape.body.position[1]
    #     xs.append(x/n)
    #     ys.append(y/n)
    #     kes.append(ke)
    # plt.plot(kes)
    # plt.show()
    #
    # plt.plot(xs, ys)
    # plt.show()

if __name__ == '__main__':
    main()
