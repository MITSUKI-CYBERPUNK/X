// ？
#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTICES 100

// 定义结构体，numVertices是顶点数，adjacencyMatrix是图的邻接矩阵
struct Graph {
    int numVertices;
    int adjacencyMatrix[MAX_VERTICES][MAX_VERTICES];
};

struct Graph* create(int vertices) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->numVertices = vertices;

    int i, j;
    for (i = 0; i < vertices; i++) {
        for (j = 0; j < vertices; j++) {
            graph->adjacencyMatrix[i][j] = 0;
        }
    }

    return graph;
}

// 添加无向边
void addEdge(struct Graph* graph, int src, int dest) {
    graph->adjacencyMatrix[src][dest] = 1;
    graph->adjacencyMatrix[dest][src] = 1;
}

// 打开邻接表
void printGraph(struct Graph* graph) {
    int i, j;
    for (i = 0; i < graph->numVertices; i++) {
        printf("Vertex %d is connected to: ", i);
        for (j = 0; j < graph->numVertices; j++) {
            if (graph->adjacencyMatrix[i][j] == 1) {
                printf("%d ", j);
            }
        }
        printf("\n");
    }
}

int main() {
    int numVertices = 5;
    struct Graph* graph = create(numVertices);

    addEdge(graph, 0, 1);
    addEdge(graph, 0, 4);
    addEdge(graph, 1, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);
    addEdge(graph, 2, 3);
    addEdge(graph, 3, 4);

    printGraph(graph);

    return 0;
}
