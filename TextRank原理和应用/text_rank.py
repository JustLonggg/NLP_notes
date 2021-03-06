#encoding=utf8
from jieba import analyse

textrank = analyse.textrank
text = '非常线程是程序执行时的最小单位，它是进程的一个执行流，\ 是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\ 线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\ 线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\ 同样多线程也可以实现并发操作，每个请求分配一个线程来处理。'
keywords = textrank(text,topK=10,withWeight=True,allowPOS=('ns','n','vn','v'))  

for key in keywords:
    print(key)