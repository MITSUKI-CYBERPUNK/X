#include <stdio.h>
#include <stdlib.h>

#define MAX_SIZE 100

// 定义结构体
struct Queue {
    int items[MAX_SIZE];
    int front;
    int rear;
};

// 初始化队列
struct Queue* create() {
    struct Queue* queue = (struct Queue*)malloc(sizeof(struct Queue));
    queue->front = -1;
    queue->rear = -1;
    return queue;
}

// 判断队列是否为空
int isEmpty(struct Queue* queue) {
    return (queue->front == -1);
}

// 判断队列是否已满
int isFull(struct Queue* queue) {
    return (queue->rear == MAX_SIZE - 1);
}

// 插入
void enqueue(struct Queue* queue, int value) {
    if (isFull(queue)) {
        printf("Queue is full\n");
    } else {
        if (isEmpty(queue)) {
            queue->front = 0;
        }
        queue->items[++queue->rear] = value;// 入队，将新元素value添加到队列的末尾。
        // queue->rear 表示当前队列的末尾位置
    }
}

// 移除
int dequeue(struct Queue* queue) {
    int item;
    if (isEmpty(queue)) {
        printf("Queue is empty\n");
        return -1;
    } else {
        item = queue->items[queue->front];
        queue->front++;
        if (queue->front > queue->rear) {
            queue->front = queue->rear = -1;
        }
        return item;
    }
}

int main() {
    struct Queue* queue = create();

    enqueue(queue, 1);
    enqueue(queue, 2);
    enqueue(queue, 3);

    printf("Dequeued item: %d\n", dequeue(queue));
    printf("Dequeued item: %d\n", dequeue(queue));

    enqueue(queue, 4);

    printf("Dequeued item: %d\n", dequeue(queue));
    printf("Dequeued item: %d\n", dequeue(queue));

    return 0;
}
