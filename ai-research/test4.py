import asyncio, time, threading
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)
# logging.basicConfig(
#     level=logging.INFO,                          # 日志级别
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 格式
#     datefmt='%Y-%m-%d %H:%M:%S',                 # 时间格式
#     filename='app.log',                           # 输出到文件（默认控制台）
#     filemode='a',                                 # 追加模式
# )
logging.basicConfig(
    level=logging.INFO,  # 设置最低级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def work(name, t):
    print(f'{name} start {threading.current_thread}')
    # 不会阻塞等待结果，而是执行其他异步
    await asyncio.sleep(t)
    # 先拿到结果（结束）的⬆先执行
    print(f'{name} end after {t}s {threading.current_thread}')


async def work2(name, t):
    print(f'{name} start {threading.current_thread}')
    # 不会阻塞等待结果，而是执行其他异步
    await asyncio.sleep(t)
    # 先拿到结果（结束）的⬆先执行
    print(f'{name} end after {t}s {threading.current_thread}')

async def main():
    start = time.time()
    await asyncio.gather(
        work('A', 3),
        work('B', 1),
        work('C', 2)
    )
    # 结果是3点几而不是6点几
    print(f'total: {time.time() - start}')

async def wget(host):
    print('wget %s...' % host)
    connect = asyncio.open_connection(host, 80)
    reader, writer = await connect
    header = 'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % host
    writer.write(header.encode('utf-8'))
    await writer.drain()
    while True:
        line = await reader.readline()
        if line == b'\r\n':
            break
        print('%s header > %s' % (host, line.decode('utf-8').rstrip()))
    # Ignore the body, close the socket
    writer.close()


async def main2():
    tasks = [wget(host) for host in [ 'www.sohu.com', 'www.163.com']]
    #把tasks列表转化成哈希的可变参数，否则报错
    await asyncio.gather(*tasks)



if __name__ == '__main__':

    # asyncio.run(main())

    # asyncio.run(main2())

    # start = time.time()
    # loop = asyncio.new_event_loop()
    # a1 = loop.create_task(work('A', 3))
    # a2 = loop.create_task(work('B', 1))
    # tasks = [a1, a2]
    # loop.run_until_complete(asyncio.wait(tasks))
    # loop.close()
    #
    # print(f'total: {time.time() - start}')
    # logging.debug("这条不会显示（级别太低）")
    # logging.info("程序启动")
    # logging.warning("警告：配置未找到")
    # logging.error("错误：数据库连接失败")
    # logging.critical("严重错误：系统崩溃")
    # c = num()
    # c.send(None)
    # c.send(None)
    # c.send(None)
    # logger.info('>' * 10, 'end', '<' * 60)

    # client = OpenAI(
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key="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"
    # )
    #
    # completion = client.chat.completions.create(
    #     model="deepseek-ai/deepseek-v4-flash",
    #     messages=[{"role": "user", "content": "2026年广东高考语文作文题目"}],
    #     temperature=1,
    #     top_p=0.95,
    #     max_tokens=16384,
    #     extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
    #     stream=False
    # )
    # print("*"*80)
    # reasoning = getattr(completion.choices[0].message, "reasoning", None) or getattr(completion.choices[0].message,
    #                                                                                  "reasoning_content", None)
    # if reasoning:
    #     print(reasoning)
    #
    #
    # print(completion.choices[0].message.content)

    from operator import itemgetter

    L = ["bob", "about", "Zoo", "Credit"]

    print(sorted(L))
    print(sorted(L, key=str.lower))

    students = [("Bob", 75), ("Adam", 92), ("Bart", 66), ("Lisa", 88)]

    print(sorted(students, key=itemgetter(0)))
    print(sorted(students, key=lambda t: t[1]))
    print(sorted(students, key=itemgetter(1), reverse=True))
