from collections import namedtuple, defaultdict
import random
import numpy as np

GRID_SIZE = 10

# Named tuple to represent point
Point = namedtuple("Point", "x, y")

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [Point(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = Point(0, 1)
        self.food = self.place_food()
        self.score = 0
        self.steps = 0

    def place_food(self):
        x, y = random.randint(1, GRID_SIZE - 1), random.randint(1, GRID_SIZE - 1)
        while Point(x, y) in self.snake:
            x, y = random.randint(1, GRID_SIZE - 1), random.randint(1, GRID_SIZE - 1)
        return Point(x, y)

    def play_step(self, action):
        if action == 0:  # Straight
            pass
        elif action == 1:  # Left turn
            self.direction = Point(-self.direction.y, self.direction.x)
        elif action == 2:  # Right turn
            self.direction = Point(self.direction.y, -self.direction.x)

        # Move snake
        new_head = Point(self.snake[0].x + self.direction.x, self.snake[0].y + self.direction.y)
        self.snake = [new_head] + self.snake[:-1]

        # Check for collisions
        if (new_head.x < 0 or new_head.x >= GRID_SIZE or
            new_head.y < 0 or new_head.y >= GRID_SIZE or
            new_head in self.snake[1:]):
            return True, self.score  # Game over

        # Check for food
        if new_head == self.food:
            self.snake.append(self.snake[-1])
            self.food = self.place_food()
            self.score += 1
            self.steps = 0  # Reset steps counter

        # Increment steps
        self.steps += 1
        return False, self.score  # Game not over

    def get_state(self):
        head = self.snake[0]
        state = [
            # Danger straight, left, right
            self.check_collision(Point(head.x + self.direction.x, head.y + self.direction.y)),
            self.check_collision(Point(head.x - self.direction.y, head.y + self.direction.x)),
            self.check_collision(Point(head.x + self.direction.y, head.y - self.direction.x)),
            # Food location relative to head
            self.food.x < head.x,  # Food left
            self.food.x > head.x,  # Food right
            self.food.y < head.y,  # Food above
            self.food.y > head.y   # Food below
        ]
        return np.array(state, dtype=int)

    def check_collision(self, point):
        return (point.x < 0 or point.x >= GRID_SIZE or
                point.y < 0 or point.y >= GRID_SIZE or
                point in self.snake)