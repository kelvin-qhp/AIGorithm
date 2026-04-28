from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def prompt_chat_template_invoke():
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题，并确保没有错误"),
        ("human", "请写一个Python程序，关于{question}")
    ])

    print(chat_prompt.invoke({"question": "冒泡排序"}))

def prompt_chat_template_format():
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题，并确保没有错误"),
        ("human", "请写一个Python程序，关于{question}")
    ])

    print(chat_prompt.format(question="冒泡排序"))

def prompt_chat_template_placeholder_invoke():
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("memory"),
        SystemMessage("你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题"),
        ("human", "{question}")
    ])

    prompt_value = prompt.invoke({"memory": [HumanMessage("我的名字叫大志，是一名程序员"),
                                             AIMessage("好的，大志你好")],
                                  "question": "请问我的名字叫什么？"})
    print(prompt_value.to_string())

def prompt_chat_template_placeholder2_invoke():
    prompt = ChatPromptTemplate.from_messages([
        ("placeholder", "{memory}"),
        SystemMessage("你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题"),
        ("human", "{question}")
    ])

    prompt_value = prompt.invoke({"memory": [HumanMessage("我的名字叫大志，是一名程序员"),
                                             AIMessage("好的，大志你好")],
                                  "question": "请问我的名字叫什么？"})
    print(prompt_value.to_string())


def prompt_chat_template_plus_invoke():
    first_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是OpenAI开发的大语言模型，下面所有提问你扮演小米雷军的角色，对我的提问进行回答")
    ])
    second_chat_prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}")
    ])

    all_chat_prompt = first_chat_prompt + second_chat_prompt
    print(all_chat_prompt.invoke({"question": "Are you OK?"}).to_string())


    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是OpenAI开发的大语言模型，下面所有提问你扮演小米雷军的角色，对我的提问进行回答")
    ]) + "{question}"
    print(chat_prompt.invoke({"question": "Are you OK?"}).to_string())


def prompt_template_standard():
    template = """  
    你是一位专业的{domain}顾问，请用{language}回答以下问题，要求简洁易懂（适合零基础）：  
    问题：{question}  
    回答：  
    """

    # 3. 创建PromptTemplate实例：指定“必须填的变量”（input_variables）
    prompt = PromptTemplate(
        input_variables=["domain", "language", "question"],  # 3个必填变量，少填会报错
        template=template  # 关联上面定义的模板
    )

    # 4. 查看模板结构（可选，帮你确认变量是否正确）
    print("模板结构：", prompt)
    # 输出会显示：input_variables=['domain', 'language', 'question'], template=...

    # 5. 用format()填充变量，生成最终提示词
    final_prompt = prompt.format(
        domain="网络安全",        # 填“领域”变量
        language="中文",         # 填“语言”变量
        question="如何防范钓鱼攻击？"  # 填“问题”变量
    )

    # 6. 查看最终提示词（实际调用大模型时，就用这个final_prompt作为输入）
    print("\n最终提示词：", final_prompt)

def prompt_template_format():
    # 1. 定义模板：里面有{target_language}和{text}两个占位符
    template = "请将以下文本翻译成{target_language}，翻译后保持原意不变：\n文本：{text}"

    # 2. 用from_template()创建实例：自动识别模板中的{变量}，生成input_variables
    prompt = PromptTemplate.from_template(template)

    # 3. 查看自动推断的变量（验证是否正确）
    print("自动识别的变量：", prompt.input_variables)  # 输出：['target_language', 'text']

    # 4. 填充变量（和案例1一样用format()）
    final_prompt = prompt.format(
        target_language="英语",
        text="LangChain的PromptTemplate很适合零基础学习"
    )

    print("\n最终提示词：", final_prompt)

def prompt_template_partial_var():
    # 1. 定义模板：包含“固定默认变量”{analysis_type}和“动态变量”{user_input}
    template = """分析用户输入的情绪（默认分析类型：{analysis_type}）：
    用户输入：{user_input}
    分析结果（要求：用“正面/负面/中性”总结，再补1句解释）："""

    # 2. 创建实例：用partial_variables固定默认值，input_variables只填“动态变量”
    prompt_template = PromptTemplate(
        input_variables=["user_input"],  # 只需要填“用户输入”这个动态变量
        template=template,
        template_format="f-string",  # 指定模板格式为f-string（适配Python字符串语法）
        partial_variables={"analysis_type": "情感极性分析"}  # 固定“分析类型”，不用每次填
    )

    # 3. 填充动态变量（只需填user_input）
    final_prompt = prompt_template.format(user_input="这个产品太难用了")
    print("最终提示词：", final_prompt)

    # 4. 查看模板的所有属性（可选，帮你理解内部结构）
    print("\n模板文本：", prompt_template.template)
    print("必填变量：", prompt_template.input_variables)
    print("默认变量：", prompt_template.partial_variables)

if __name__ == '__main__':
    prompt_chat_template_invoke()
    prompt_chat_template_format()
    prompt_chat_template_placeholder_invoke()
    prompt_chat_template_placeholder2_invoke()
    prompt_chat_template_plus_invoke()
    # prompt_template_standard()
    # prompt_template_format()
    # prompt_template_partial_var()