import turtle


class PingPong():
    def __init__(self):
        self.done = False

        self.reward = 0

        self.win = turtle.Screen()    # Create a screen
        self.win.title('Paddle')      # Set the title to paddle
        self.win.bgcolor('black')     # Set the color to black
        self.win.tracer(0)
        # Set the width and height to 600
        self.win.setup(width=600, height=600)

        # Paddle
        self.paddle = turtle.Turtle()    # Create a turtle object
        self.paddle.shape('square')      # Select a square shape
        self.paddle.speed(0)
        # Streach the length of square by 5
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.penup()
        self.paddle.color('white')       # Set the color to white
        # Place the shape on bottom of the screen
        self.paddle.goto(0, -275)

        # Ball
        self.ball = turtle.Turtle()      # Create a turtle object
        self.ball.speed()
        self.ball.shape('circle')        # Select a circle shape
        self.ball.color('red')           # Set the color to red
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 10   # self.ball's x-axis velocity
        self.ball.dy = -10  # ball's y-axis velocity

        self.hit, self.miss = 0, 0

        # Scorecard
        self.score = turtle.Turtle()   # Create a turtle object
        self.score.speed(0)
        self.score.color('white')      # Set the color to white
        self.score.hideturtle()        # Hide the shape of the object
        # Set scorecard to upper middle of the screen
        self.score.goto(0, 250)
        self.score.penup()
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss),
                         align='center', font=('Courier', 24, 'normal'))

        # Keyboard Control
        self.win.listen()
        # call paddle_right on right arrow key
        self.win.onkey(self.paddle_right, 'Right')
        # call paddle_left on right arrow key
        self.win.onkey(self.paddle_left, 'Left')

    def paddle_right(self):
        x = self.paddle.xcor()        # Get the x position of self.paddle
        if x < 225:
            self.paddle.setx(x+20)    # increment the x position by 20

    def paddle_left(self):
        x = self.paddle.xcor()        # Get the x position of self.paddle
        if x > -225:
            self.paddle.setx(x-20)    # decrement the x position by 20

    def run_frame(self):
        self.win.update()

        # Ball moving

        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision

        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact

        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(
                self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # Ball Paddle collision

        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(
                self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3

 # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):

        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

    def step(self, action):

        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()
            self.reward += .1

        if action == 2:
            self.paddle_right()
            self.reward += .1

        self.run_frame()

        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01,
                 self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        return self.reward, state, self.done


# env = PingPong()

# while True:
#     env.run_frame()
