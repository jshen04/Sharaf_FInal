import pymunk
import pygame
import pymunk.pygame_util
import math
from pygame.locals import USEREVENT, QUIT, KEYDOWN, KEYUP, K_s, K_r, K_q, K_ESCAPE, K_UP, K_DOWN
import numpy as np

BALL_MASS = 1
BALL_RADIUS = 25
BALL_FRICTION = 1
BALL_ELASTICITY = 1

WORLD_DIMS = (1200, 600)
GATE_ELASTICITY = 1
GATE_FRICTION = 1

class Ball:
    def __init__(self, position, collision_type=1):
        ball_body = pymunk.Body(mass=BALL_MASS, moment=pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
        ball_body.position = position
        ball_shape = pymunk.Circle(ball_body, radius=BALL_RADIUS)
        ball_shape.friction = BALL_FRICTION
        ball_shape.elasticity = BALL_ELASTICITY
        ball_shape.collision_type = collision_type
        self.shape = ball_shape

    def set_position(self, position):
        self.shape.body.position = self.set_position(position)

class CueBall(Ball):
    def __init__(self, position, collision_type=2):
        super().__init__(position, collision_type=collision_type)

def initialize_border(space):
    t = pymunk.Body(body_type=pymunk.Body.STATIC)
    t.elasticity = GATE_ELASTICITY
    t.friction = GATE_FRICTION
    t.position = (WORLD_DIMS[0]/2, WORLD_DIMS[1])
    ts = pymunk.Poly.create_box(body=t, size=(WORLD_DIMS[0], 25))
    ts.collision_type = 3

    b = pymunk.Body(body_type=pymunk.Body.STATIC)
    b.elasticity = GATE_ELASTICITY
    b.friction = GATE_FRICTION
    b.position = (WORLD_DIMS[0]/2, 0)
    bs = pymunk.Poly.create_box(body=b, size=(WORLD_DIMS[0], 25))
    bs.collision_type = 3

    l = pymunk.Body(body_type=pymunk.Body.STATIC)
    l.elasticity = GATE_ELASTICITY
    l.friction = GATE_FRICTION
    l.position = (0, WORLD_DIMS[1]/2)
    ls = pymunk.Poly.create_box(body=l, size=(25, WORLD_DIMS[1]))
    ls.collision_type = 4

    r = pymunk.Body(body_type=pymunk.Body.STATIC)
    r.elasticity = GATE_ELASTICITY
    r.friction = GATE_FRICTION
    r.position = (WORLD_DIMS[0], WORLD_DIMS[1]/2)
    rs = pymunk.Poly.create_box(body=r, size=(25, WORLD_DIMS[1]))
    rs.collision_type = 4

    space.add(ts, bs, ls, rs)
    space.add(t, b, l, r)


def initialize_holes(space):
    for i in range(6):
        hole = bot_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        hole.position = (WORLD_DIMS[0] * ((i % 3)/2), WORLD_DIMS[1] * (i//3))
        hole_shape = pymunk.Circle(bot_body, radius=BALL_RADIUS * 1.5)
        space.add(hole_shape)

def main():
    pygame.init()
    screen = pygame.display.set_mode(WORLD_DIMS)
    clock = pygame.time.Clock()
    running = True
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    space = pymunk.Space(WORLD_DIMS)
    space.gravity = 0, 0
    initialize_border(space)
    initialize_holes(space)

    for i in range(1, 6, 1):
        for j in range(1, i + 1, 1):
            print(i, j)
            x = (3 - i) * BALL_RADIUS * 2 + 300
            y = (j - 3) * BALL_RADIUS * 2 + 300 + ((5 - i) * BALL_RADIUS)
            # x = ((j % 5) - (5 / 2) + 0.5) * BALL_RADIUS * 2 + 300
            # y = ((j // 5) - (5 / 2) + 0.5) * BALL_RADIUS * 2 + 300
            ball = Ball(position=(x, y))
            space.add(ball.shape, ball.shape.body)

    def elastic_collision(arbiter, space, data):
        s1 = arbiter.shapes[0]
        s2 = arbiter.shapes[1]
        b1 = s1.body
        b2 = s2.body
        m1 = b1.mass
        m2 = b2.mass
        v1x, v1y = b1.velocity
        v2x, v2y = b2.velocity

        v1xf = ((m1 - m2) / (m1 + m2)) * v1x + ((2 * m2) / (m1 + m2)) * v2x
        v2xf = ((2 * m1) / (m1 + m2)) * v1x - ((m1 - m2) / (m1 + m2)) * v2x

        v1yf = ((m1 - m2) / (m1 + m2)) * v1y + ((2 * m2) / (m1 + m2)) * v2y
        v2yf = ((2 * m1) / (m1 + m2)) * v1y - ((m1 - m2) / (m1 + m2)) * v2y

        b1.velocity = pymunk.Vec2d(v1xf, v1yf)
        b2.velocity = pymunk.Vec2d(v2xf, v2yf)
        return True

    def wall_collision_top_bottom(arbiter, space, body):
        s1 = arbiter.shapes[0]
        b1 = s1.body
        v1x, v1y = b1.velocity
        b1.velocity = pymunk.Vec2d(v1x, -1 * v1y)
        return True

    def wall_collision_left_right(arbiter, space, body):
        s1 = arbiter.shapes[0]
        b1 = s1.body
        v1x, v1y = b1.velocity
        b1.velocity = pymunk.Vec2d(-1 * v1x, v1y)
        return True

    ball_collision = space.add_collision_handler(1, 2)
    ball_collision.pre_solve = elastic_collision

    ball_collision = space.add_collision_handler(1, 1)
    ball_collision.pre_solve = elastic_collision

    ball_collision = space.add_collision_handler(1, 3)
    ball_collision.pre_solve = wall_collision_top_bottom

    ball_collision = space.add_collision_handler(1, 4)
    ball_collision.pre_solve = wall_collision_left_right

    ball_collision = space.add_collision_handler(2, 3)
    ball_collision.pre_solve = wall_collision_top_bottom

    ball_collision = space.add_collision_handler(2, 4)
    ball_collision.pre_solve = wall_collision_left_right

    cueball = CueBall(position=(1000, 300))
    space.add(cueball.shape, cueball.shape.body)
    cueball.shape.body.apply_impulse_at_local_point((-250, 0))


    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key in (K_q, K_ESCAPE)):
                running = False
            elif event.type == KEYDOWN and event.key == K_s:
                # Start/stop simulation.
                running = not running

        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        pygame.display.update()
        space.step(0.01)

if __name__ == '__main__':
    main()

