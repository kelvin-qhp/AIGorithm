from manim import *
import numpy as np


class DetailedTransformer3Layers(Scene):
    def construct(self):
        # 标题
        title = Text("3-Layer Transformer - Detailed View", font_size=40, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # 创建三层
        layers = VGroup()
        layer_colors = [BLUE, PURPLE, ORANGE]
        layer_names = ["Layer 1", "Layer 2", "Layer 3"]

        for i in range(3):
            layer = self.create_detailed_layer(
                layer_names[i],
                layer_colors[i],
                x_pos=-4 + i * 4,
                y_pos=0
            )
            layers.add(layer)
            self.play(Create(layer), run_time=1.5)

            # 添加层间连接
            if i > 0:
                arrow = Arrow(
                    layers[i - 1].get_right(),
                    layers[i].get_left(),
                    buff=0.3,
                    color=layer_colors[i],
                    stroke_width=3
                )
                self.play(Create(arrow))

        self.wait(1)

        # 添加输入和输出
        input_label = Text("Input", font_size=28, color=WHITE)
        input_label.next_to(layers[0], LEFT, buff=1.5)
        self.play(Write(input_label))

        output_label = Text("Output", font_size=28, color=WHITE)
        output_label.next_to(layers[2], RIGHT, buff=1.5)
        self.play(Write(output_label))

        # 添加数据流动画
        self.animate_data_flow(layers)

        # 显示总结
        summary = VGroup(
            Text("Transformer Stack:", font_size=32, color=YELLOW),
            Text("• Multi-Head Attention", font_size=24, color=TEAL),
            Text("• Feed-Forward Network", font_size=24, color=GREEN),
            Text("• Residual Connections", font_size=24, color=YELLOW),
            Text("• Layer Normalization", font_size=24, color=PINK),
        ).arrange(DOWN, aligned_edge=LEFT)
        summary.to_edge(DOWN, buff=0.5)

        self.play(Write(summary))
        self.wait(3)

    def create_detailed_layer(self, name, color, x_pos=0, y_pos=0):
        """创建详细的层结构"""
        layer = VGroup()

        # 层背景
        background = Rectangle(
            width=2.8,
            height=4.5,
            color=color,
            fill_opacity=0.1,
            stroke_width=2
        )
        background.move_to([x_pos, y_pos, 0])

        # 层标题
        title = Text(name, font_size=22, color=color)
        title.next_to(background, UP, buff=0.1)

        # 组件
        components = [
            ("Multi-Head\nAttention", TEAL, 0.8),
            ("Add & Norm", YELLOW, 0.5),
            ("Feed-Forward\nNetwork", GREEN, 0.8),
            ("Add & Norm", YELLOW, 0.5),
        ]

        comp_group = VGroup()
        y_offset = 1.5
        for comp_text, comp_color, height in components:
            rect = Rectangle(
                width=2.0,
                height=height,
                color=comp_color,
                fill_opacity=0.2,
                stroke_width=2
            )
            label = Text(comp_text, font_size=16, color=comp_color)
            label.move_to(rect.get_center())
            comp = VGroup(rect, label)
            comp.move_to([x_pos, y_offset, 0])
            comp_group.add(comp)
            y_offset -= (height + 0.3)

            # 添加箭头连接
            if len(comp_group) > 1:
                arrow = Arrow(
                    comp_group[-2].get_bottom(),
                    comp_group[-1].get_top(),
                    buff=0.1,
                    color=WHITE,
                    stroke_width=2
                )
                comp_group.add(arrow)

        layer.add(background, title, comp_group)
        return layer

    def animate_data_flow(self, layers):
        """动画展示数据流"""
        # 创建光点
        dot = Dot(color=RED, radius=0.1)
        dot.move_to(layers[0].get_left() + LEFT * 0.3)

        # 路径：从左到右穿过所有层
        for layer in layers:
            # 进入层
            self.play(
                dot.animate.move_to(layer.get_center()),
                run_time=0.8
            )
            # 在层内移动（模拟计算）
            self.play(
                dot.animate.shift(UP * 0.5),
                run_time=0.3
            )
            self.play(
                dot.animate.shift(DOWN * 0.5),
                run_time=0.3
            )

        # 移动到输出
        self.play(
            dot.animate.move_to(layers[-1].get_right() + RIGHT * 0.3),
            run_time=0.8
        )
        self.play(FadeOut(dot))