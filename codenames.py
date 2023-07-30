from playgroundrl.client import *
from playgroundrl.actions import *
from word2vec import *

BOARD_SIZE = 25

class TestCodenames(PlaygroundClient):
    def __init__(self):
        super().__init__(
            GameType.CODENAMES,
            model_name="rohan-codenames",
            auth={
                "email": "rohankrishnan00@gmail.com",
                "api_key": "N038MxaxBpktIgGP40Yk3u1rmm2FP7x0THpPufnsHsM",
            },
            render_gameplay=True,
        )

    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            return None

        if state.role == "GIVER":
            word, count = clue(state)
            return CodenamesSpymasterAction(
                word = word, 
                count = count 
            )
        elif state.role == "GUESSER":
            return CodenamesGuesserAction(
                guesses= guess(state),
            )

    def gameover_callback(self):
        pass

if __name__ == "__main__":
    # args = parse_arguments("codenames")
    t = TestCodenames()
    t.run(
        # pool=Pool(),
        # num_games=args.num_games,
        # self_training=True,
        # maximum_messages=500000,
        # used to set up 2-player game (rather than default 4)
        # game_parameters={"num_players": 2},
    )