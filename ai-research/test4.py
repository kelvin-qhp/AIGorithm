import asyncio, time, threading


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

    start = time.time()
    loop = asyncio.new_event_loop()
    a1 = loop.create_task(work('A', 3))
    a2 = loop.create_task(work('B', 1))
    tasks = [a1, a2]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    print(f'total: {time.time() - start}')