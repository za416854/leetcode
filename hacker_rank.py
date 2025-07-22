import heapq
from collections import deque
from typing import List, Optional


class Solution:
    def flippingMatrix(matrix):
        # 題目給的是一個 2n × 2n 的正方形矩陣（例如 4x4、6x6、8x8），你要最大化的是左上角 n × n 的總和，所以我們要知道 n 的值
        n = len(matrix) / 2
        total = 0
        for i in range(n):
            for j in range(n):
                total = max(  # 左上角每一格 (i, j) 翻轉後能拿到的最大值，這些最大值的「總和」。
                    # 這裡是對 每格 做翻轉，而不是左上角四格或六格，不要搞錯
                    matrix[i][j],
                    matrix[i][n * 2 - j - 1],
                    matrix[n * 2 - i - 1][j],
                    matrix[n * 2 - i - 1][n * 2 - j - 1],
                )
        return total
