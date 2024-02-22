#include <stdio.h>
#include <stdlib.h>

// 定义结构体
struct Node {
    int data;
    struct Node* next;
};

// 头插法
void insert(struct Node** head, int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = *head;// 新节点指向原始的头结点指向的结点
    *head = newNode;// 头指针指向新插入的节点
}

// 删除
void delete(struct Node** head, int key) {
    struct Node *temp = *head, *prev = NULL;

    if (temp != NULL && temp->data == key) {
        *head = temp->next;
        free(temp);
        return;
    }

//遍历
    while (temp != NULL && temp->data != key) {
        prev = temp;
        temp = temp->next;
    }

    if (temp == NULL) return;

    prev->next = temp->next;
    free(temp);
}

void print(struct Node* node) {
    while (node != NULL) {
        printf("%d -> ", node->data);
        node = node->next;
    }
    printf("NULL\n");
}

int main() {
    struct Node* head = NULL;

    insert(&head, 3);
    insert(&head, 2);
    insert(&head, 1);

    printf("\n");
    print(head);

    delete(&head, 2);
    print(head);

    return 0;
}