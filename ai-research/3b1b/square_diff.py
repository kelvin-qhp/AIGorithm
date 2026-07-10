from manim import *
import numpy as np

'''
3b1b:
    manim -pql square_diff.py SquareDifference
'''
class SquareDifference(Scene):
    def construct(self):
        a = 3
        b = 1
        side_length = a

        big_square = Square(side_length=side_length, color=BLUE)
        big_square.shift(LEFT * 3)

        small_square = Square(side_length=b, color=RED)
        small_square.move_to(big_square.get_corner(DR) - np.array([b / 2, b / 2, 0]))

        a_label = MathTex("a", color=BLUE).next_to(big_square.get_top(), UP, buff=0.1)
        b_label = MathTex("b", color=RED).next_to(small_square.get_right(), RIGHT, buff=0.1)

        self.play(Create(big_square), Write(a_label))
        self.wait(1)
        self.play(Create(small_square), Write(b_label))
        self.wait(1)

        # 方法1：使用 Polygon（推荐）
        l_shape = Polygon(
            big_square.get_corner(UL),
            big_square.get_corner(UR),
            big_square.get_corner(DR) - np.array([b, 0, 0]),
            big_square.get_corner(DR) - np.array([b, b, 0]),
            big_square.get_corner(DL) + np.array([0, b, 0]),
            color=YELLOW,
            fill_opacity=0.4
        )

        self.play(FadeIn(l_shape))
        self.wait(1)

        formula = MathTex(
            "a^2 - b^2 = (a+b)(a-b)",
            color=WHITE
        )
        formula.to_edge(DOWN)
        self.play(Write(formula))
        self.wait(2)

        rect1 = Rectangle(width=a - b, height=a, color=GREEN, fill_opacity=0.3)
        rect1.move_to(big_square.get_center() + np.array([-(b / 2), 0, 0]))

        rect2 = Rectangle(width=b, height=a - b, color=GREEN, fill_opacity=0.3)
        rect2.move_to(big_square.get_center() + np.array([(a - b) / 2, b / 2, 0]))

        self.play(
            Transform(l_shape.copy(), rect1),
            Transform(l_shape.copy(), rect2),
            run_time=2
        )
        self.wait(2)

        target_rect = Rectangle(width=a + b, height=a - b, color=GREEN, fill_opacity=0.3)
        target_rect.shift(RIGHT * 3)

        self.play(
            rect1.animate.move_to(target_rect.get_left() + np.array([(a - b) / 2, 0, 0])),
            rect2.animate.move_to(target_rect.get_right() - np.array([(a - b) / 2, 0, 0])),
            run_time=2
        )
        self.wait(1)

        conclusion = MathTex(
            "a^2 - b^2 = (a+b)(a-b)",
            color=YELLOW
        ).scale(1.5).to_edge(UP)

        self.play(Transform(formula, conclusion))
        self.wait(3)