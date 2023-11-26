from manim import *
import numpy as np

class demdNumbers(Scene):
    def construct(self):
        a1 = [[0.50], [0.30], [0.10], [0.05], [0.05]]
        a2 = [[0.10, 0.15, 0.25, 0.30, 0.20]]
        dpmat = [[np.random.rand()]*5 for _ in range(5)]

        m0 = DecimalMatrix(dpmat,element_to_mobject_config={"num_decimal_places": 2})
        v1 = DecimalMatrix(a1,element_to_mobject_config={"num_decimal_places": 2})
        v2 = DecimalMatrix(a2,element_to_mobject_config={"num_decimal_places": 2})

        v1.next_to(m0, LEFT)
        v2.next_to(m0, UP)

        #self.add(m0)
        #self.add(v1)
        #self.add(v2)


        img = ImageMobject(dpmat).scale(100).shift(0)
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.add(img)

        chart = BarChart(
            a2[0],
            y_range=[0, 1.0, 2],
            y_axis_config={"font_size": 24},
        )
        self.add(chart)

        chart.change_bar_values(list(reversed(a2[0])))
        self.add(chart.get_bar_labels(font_size=24))

        #dpmat[2][2] = 0.5
        #img1 = ImageMobject(dpmat).shift(0)
        #m1 = DecimalMatrix(dpmat,element_to_mobject_config={"num_decimal_places": 2})
        #self.play(ReplacementTransform(img, img1), ReplacementTransform(m0,m1))
        #self.play(ReplacementTransform(m0, m1))

        # v1img = ImageMobject(v1)
        # v1img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        # self.add(v1img)
        # v2img = ImageMobject(v2)
        # v2img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        # self.add(v2img)



class MatrixExamples(Scene):
    def construct(self):
        m0 = Matrix([[2, "\pi"], [-1, 1]])
        m1 = Matrix([[2, 0, 4], [-1, 1, 5]],
            v_buff=1.3,
            h_buff=0.8,
            bracket_h_buff=SMALL_BUFF,
            bracket_v_buff=SMALL_BUFF,
            left_bracket="\{",
            right_bracket="\}")
        m1.add(SurroundingRectangle(m1.get_columns()[1]))
        m2 = Matrix([[2, 1], [-1, 3]],
            element_alignment_corner=UL,
            left_bracket="(",
            right_bracket=")")
        m3 = Matrix([[2, 1], [-1, 3]],
            left_bracket="\\langle",
            right_bracket="\\rangle")
        m4 = Matrix([[2, 1], [-1, 3]],
        ).set_column_colors(RED, GREEN)
        m5 = Matrix([[2, 1], [-1, 3]],
        ).set_row_colors(RED, GREEN)
        g = Group(
            m0,m1,m2,m3,m4,m5
        ).arrange_in_grid(buff=2)
        self.add(g)
