#include <stdio.h>
#include <stdlib.h>

// 初始化结构体
struct Node {
    int data;
    struct Node* left;
    struct Node* right;
};

// 创建新节点
struct Node* create(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

// 中序遍历：左子树-根节点-右子树
void inorderTraversal(struct Node* root) {
    if (root != NULL) {
        inorderTraversal(root->left);
        printf("%d ", root->data);
        inorderTraversal(root->right);
    }
}

int main() {
    struct Node* root = create(1);
    root->left = create(2);
    root->right = create(3);
    root->left->left = create(4);
    root->left->right = create(5);

    inorderTraversal(root);
    printf("\n");

    return 0;
}
