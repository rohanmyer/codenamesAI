from playgroundrl.client import *

class TestSnake(PlaygroundClient):
    def __init__(self):
        super().__init__(
            game=GameType.SNAKE,
            model_name="tutorial-snake",
            auth={
                "email": "rohankrishnan00@gmail.com",
                "api_key": "N038MxaxBpktIgGP40Yk3u1rmm2FP7x0THpPufnsHsM",
            },
            render_gameplay=True,
        )

    def callback(self, state: SnakeState, reward):
        apple = state.apple
        snake = state.snake
        head = snake[-1]

        SIZE = 10
        x, y = head

        if x == SIZE - 1 and y == SIZE - 2:
            return "S"

        if x == 0 and y == SIZE - 1:
            return "N"

        if y == SIZE - 1:
            return "W"

        if x % 2 == 0:
            if y == 0:
                return "E"
            return "N"
        else:
            if y == SIZE - 2:
                return "E"
            return "S"
    
    def gameover_callback(self):
        pass

if __name__ == "__main__":
    t = TestSnake()
    t.run()