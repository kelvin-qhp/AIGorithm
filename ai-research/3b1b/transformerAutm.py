from manim import *
import numpy as np


class ThreeLayerTransformer(Scene):
    def construct(self):
        # 设置标题
        title = Text("3-Layer Transformer Architecture", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # ==================== 第一层：输入层 ====================
        input_label = Text("Input", font_size=30, color=BLUE)
        input_label.move_to(LEFT * 5.5 + UP * 2.5)
        self.play(Write(input_label))

        # 创建输入 token 序列 (3个token)
        input_tokens = self.create_token_sequence(3, "Input Tokens", BLUE)
        input_tokens.move_to(LEFT * 5 + UP * 2)
        self.play(Create(input_tokens))
        self.wait(0.5)

        # ==================== 第一层：Multi-Head Attention ====================
        self.mha1 = self.create_block("Multi-Head\nAttention", TEAL, 1.8)
        self.mha1.next_to(input_tokens, DOWN, buff=0.8)
        self.play(Create(self.mha1))

        # 箭头：输入 → MHA
        arrow1 = Arrow(
            input_tokens.get_bottom(),
            self.mha1.get_top(),
            buff=0.2,
            color=WHITE
        )
        self.play(Create(arrow1))
        self.wait(0.3)

        # ==================== 第一层：Add & Norm ====================
        add_norm1 = self.create_small_block("Add & Norm", YELLOW, 1.2)
        add_norm1.next_to(self.mha1, DOWN, buff=0.6)
        self.play(Create(add_norm1))

        # 残差连接 (跳过连接)
        self.skip1 = self.create_skip_connection(input_tokens, add_norm1)
        self.play(Create(self.skip1))

        # 箭头：MHA → Add & Norm
        arrow2 = Arrow(
            self.mha1.get_bottom(),
            add_norm1.get_top(),
            buff=0.2,
            color=WHITE
        )
        self.play(Create(arrow2))
        self.wait(0.3)

        # ==================== 第一层：Feed Forward ====================
        self.ff1 = self.create_block("Feed Forward", GREEN, 1.8)
        self.ff1.next_to(add_norm1, DOWN, buff=0.6)
        self.play(Create(self.ff1))

        # 箭头：Add & Norm → FF
        arrow3 = Arrow(
            add_norm1.get_bottom(),
            self.ff1.get_top(),
            buff=0.2,
            color=WHITE
        )
        self.play(Create(arrow3))
        self.wait(0.3)

        # ==================== 第一层：Add & Norm ====================
        add_norm2 = self.create_small_block("Add & Norm", YELLOW, 1.2)
        add_norm2.next_to(self.ff1, DOWN, buff=0.6)
        self.play(Create(add_norm2))

        # 残差连接
        self.skip2 = self.create_skip_connection(add_norm1, add_norm2)
        self.play(Create(self.skip2))

        # 箭头：FF → Add & Norm
        arrow4 = Arrow(
            self.ff1.get_bottom(),
            add_norm2.get_top(),
            buff=0.2,
            color=WHITE
        )
        self.play(Create(arrow4))
        self.wait(0.3)

        # ==================== 层间连接 ====================
        # 从第一层到第二层的箭头
        layer_arrow1 = Arrow(
            add_norm2.get_bottom() + DOWN * 0.3,
            DOWN * 1.5,
            buff=0.2,
            color=PURPLE,
            stroke_width=3
        )
        layer_arrow1.next_to(add_norm2, DOWN, buff=0.3)
        self.play(Create(layer_arrow1))
        self.wait(0.3)

        # ==================== 第二层 (重复第一层结构) ====================
        layer2_label = Text("Layer 2", font_size=28, color=PURPLE)
        layer2_label.move_to(RIGHT * 0 + UP * 0.5)
        self.play(Write(layer2_label))

        # 简化展示第二层
        layer2_block = self.create_layer_block("Layer 2\n(Attention + FFN)", PURPLE, 2.0)
        layer2_block.move_to(RIGHT * 0 + DOWN * 1.8)
        self.play(Create(layer2_block))

        # 从第一层到第二层的箭头
        arrow_layer1_2 = Arrow(
            layer_arrow1.get_bottom(),
            layer2_block.get_top(),
            buff=0.2,
            color=PURPLE
        )
        self.play(Create(arrow_layer1_2))
        self.wait(0.5)

        # ==================== 第三层 (重复) ====================
        layer3_label = Text("Layer 3", font_size=28, color=ORANGE)
        layer3_label.move_to(RIGHT * 4.5 + UP * 0.5)
        self.play(Write(layer3_label))

        layer3_block = self.create_layer_block("Layer 3\n(Attention + FFN)", ORANGE, 2.0)
        layer3_block.move_to(RIGHT * 4.5 + DOWN * 1.8)
        self.play(Create(layer3_block))

        # 从第二层到第三层的箭头
        arrow_layer2_3 = Arrow(
            layer2_block.get_right(),
            layer3_block.get_left(),
            buff=0.2,
            color=ORANGE
        )
        self.play(Create(arrow_layer2_3))
        self.wait(0.5)

        # ==================== 输出层 ====================
        output_label = Text("Output", font_size=30, color=RED)
        output_label.next_to(layer3_block, DOWN, buff=0.8)
        self.play(Write(output_label))

        output_tokens = self.create_token_sequence(3, "Output Tokens", RED)
        output_tokens.next_to(layer3_block, DOWN, buff=1.2)
        self.play(Create(output_tokens))

        # 箭头：第三层 → 输出
        arrow_output = Arrow(
            layer3_block.get_bottom(),
            output_tokens.get_top(),
            buff=0.2,
            color=RED
        )
        self.play(Create(arrow_output))
        self.wait(0.5)

        # ==================== 添加数据流动画 ====================
        # 用光点展示数据流
        self.show_data_flow(input_tokens, output_tokens)

        # ==================== 添加组件标签 ====================
        self.add_component_labels()

        # 总结
        summary = Text(
            "Transformer: 3 Layers of Attention & Feed-Forward",
            font_size=32,
            color=YELLOW
        )
        summary.to_edge(DOWN)
        self.play(Write(summary))
        self.wait(2)

        # 高亮显示关键组件（现在 self.mha1, self.ff1 等已定义）
        self.highlight_key_components()
        self.wait(2)

    def create_token_sequence(self, num_tokens, label, color):
        """创建 token 序列"""
        tokens = VGroup()
        for i in range(num_tokens):
            token = Square(side_length=0.5, color=color, fill_opacity=0.3)
            token_label = Text(f"x{i + 1}", font_size=20, color=color)
            token_label.move_to(token.get_center())
            token_group = VGroup(token, token_label)
            tokens.add(token_group)
        tokens.arrange(RIGHT, buff=0.3)
        return tokens

    def create_block(self, text, color, width=1.8):
        """创建矩形块"""
        rect = Rectangle(
            width=width,
            height=0.8,
            color=color,
            fill_opacity=0.2,
            stroke_width=2
        )
        label = Text(text, font_size=20, color=color)
        label.move_to(rect.get_center())
        block = VGroup(rect, label)
        return block

    def create_small_block(self, text, color, width=1.2):
        """创建小矩形块"""
        rect = Rectangle(
            width=width,
            height=0.5,
            color=color,
            fill_opacity=0.2,
            stroke_width=2
        )
        label = Text(text, font_size=16, color=color)
        label.move_to(rect.get_center())
        block = VGroup(rect, label)
        return block

    def create_layer_block(self, text, color, width=2.0):
        """创建层块"""
        rect = Rectangle(
            width=width,
            height=1.2,
            color=color,
            fill_opacity=0.15,
            stroke_width=3
        )
        label = Text(text, font_size=22, color=color)
        label.move_to(rect.get_center())
        block = VGroup(rect, label)
        return block

    def create_skip_connection(self, source, target):
        """创建残差连接（跳过连接）"""
        # 创建一条从 source 到 target 的曲线路径
        skip = DashedLine(
            source.get_left() + LEFT * 0.5,
            target.get_left() + LEFT * 0.5,
            color=YELLOW,
            stroke_width=2
        )
        return skip

    def show_data_flow(self, start, end):
        """展示数据流动画"""
        # 创建一串光点从输入流到输出
        dot = Dot(color=RED, radius=0.08)
        dot.move_to(start.get_center())

        # 路径：从输入到输出的大致路径
        path = [
            start.get_center() + UP * 0.3,
            start.get_center() + DOWN * 1.5,
            start.get_center() + DOWN * 3.0,
            end.get_center() + UP * 0.3,
        ]

        # 让光点沿路径移动
        for point in path:
            self.play(
                dot.animate.move_to(point),
                run_time=0.5
            )
        self.play(FadeOut(dot))

    def add_component_labels(self):
        """添加组件说明标签"""
        labels = VGroup(
            Text("Q, K, V", font_size=18, color=TEAL).move_to(LEFT * 4 + DOWN * 0.5),
            Text("Residual", font_size=18, color=YELLOW).move_to(LEFT * 4 + DOWN * 1.5),
            Text("FFN", font_size=18, color=GREEN).move_to(LEFT * 4 + DOWN * 2.5),
        )
        self.play(Write(labels))

    def highlight_key_components(self):
        """高亮关键组件"""
        # 高亮多头注意力
        attention_highlight = SurroundingRectangle(
            self.mha1,
            color=TEAL,
            buff=0.2,
            stroke_width=3
        )
        self.play(Create(attention_highlight))
        self.wait(0.5)

        # 高亮前馈网络
        ff_highlight = SurroundingRectangle(
            self.ff1,
            color=GREEN,
            buff=0.2,
            stroke_width=3
        )
        self.play(Create(ff_highlight))
        self.wait(0.5)

        # 高亮残差连接
        self.play(
            self.skip1.animate.set_color(YELLOW),
            self.skip2.animate.set_color(YELLOW),
            run_time=0.5
        )
        self.wait(0.5)