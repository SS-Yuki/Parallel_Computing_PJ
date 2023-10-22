#include <omp.h>
#include <stdio.h>

int main() {
// 这样用大括号括起来的语句会多线程并行执行
// 线程数量会根据机器配置自动生成
// 因此这段话会乱序输出一系列hello
#pragma omp parallel
  {
    // 获取线程id函数
    int tid = omp_get_thread_num();
    // 获取线程数量
    int num = omp_get_num_threads();
    printf("threads_num: %d, Id: %d, Hello\n", num, tid);
  }

  // 可以用以下方式手动设置线程数量
  // 设置线程数量为8， 根据机器配置自行设置
  omp_set_num_threads(8);

// 或者这样设置线程数量
// 下面这段话代表大括号内的语句会由7个线程并行执行
// 因此会乱序输出7次
#pragma omp parallel num_threads(7)
  {
    // 获取线程id函数
    int tid = omp_get_thread_num();
    printf("Id: %d, Hello\n", tid);
  }

// 这句话能让紧跟着他的for循环并行执行
#pragma omp parallel for
  for (int i = 0; i < 16; ++i) {
    int tid = omp_get_thread_num();
    printf("Loop: Id: %d, Hello\n", tid);
  }

// 同样可以这样控制线程数量
#pragma omp parallel for num_threads(5)
  for (int i = 0; i < 16; ++i) {
    int tid = omp_get_thread_num();
    printf("5 threads Loop: Id: %d, Hello\n", tid);
  }

// 这样控制多个语句块之间并行执行
// 下面代码表示两个section并行执行
// 且每个section只执行一次
#pragma omp parallel sections
  {
#pragma omp section
    {
      int tid = omp_get_thread_num();
      printf("tid: %d\n", tid);
    }
#pragma omp section
    {
      int tid = omp_get_thread_num();
      printf("tid: %d\n", tid);
    }
  }

// 三种线程与核绑定的方式
// master 代表所有线程都和最开始的主线程在同一个核上运行
// close 代表从最开始的核开始按顺序一个核一个线程分配过去
// spread 假设有m个线程，n个核，那么每个核上运行 m / n 个线程
#pragma omp proc_bind(master)
#pragma omp proc_bind(close)
#pragma omp proc_bind(spread)

  return 0;
}