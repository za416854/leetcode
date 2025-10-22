from collections import defaultdict, deque
from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 199. Binary Tree Right Side View
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        result = []
        queue = deque([root])
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if i == level_size - 1:
                    # If it's the last node in the level, means it's this is the rightmost node, add val to result
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return result

    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue = deque([root])
        max_num = float("-inf")
        res = 0
        counter = 0
        while queue:
            sum = 0
            counter += 1
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                sum += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if sum > max_num:
                max_num = sum
                res = counter
        return res

    def orangesRotting(self, grid: List[List[int]]) -> int:
        res_min = 0
        queue = deque()
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        fresh_oranges = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    fresh_oranges += 1
                if grid[i][j] == 2:
                    queue.append((i, j))
        if fresh_oranges == 0:
            return 0
        while queue:

            res_min += 1
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < len(grid)
                        and 0 <= nc < len(grid[0])
                        and grid[nr][nc] == 1
                    ):
                        grid[nr][nc] = 2
                        fresh_oranges -= 1
                        queue.append((nr, nc))
        return res_min - 1 if fresh_oranges == 0 else -1

    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        queue = deque()
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        queue.append((entrance[0], entrance[1], 0))
        maze[entrance[0]][entrance[1]] = "+"
        # paths = 0
        while queue:
            # r, c = entrance[0], entrance[1]
            r, c, steps = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < len(maze) and 0 <= nc < len(maze[0])):
                    continue
                if maze[nr][nc] == "+":
                    continue

                if nr == 0 or nr == len(maze) - 1 or nc == 0 or nc == len(maze[0]) - 1:
                    return steps + 1
                maze[nr][nc] = "+"
                queue.append((nr, nc, steps + 1))
        return -1

    # 872. Leaf-Similar Trees
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def DFS(node: Optional[TreeNode], leaves: List) -> None:
            if not node:
                return
            if not node.left and not node.right:
                leaves.append(node.val)
            DFS(node.left, leaves)
            DFS(node.right, leaves)

        leaves1 = []
        leaves2 = []
        DFS(root1, leaves1)
        DFS(root2, leaves2)
        return leaves1 == leaves2

    # 104. Maximum Depth of Binary Tree
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1

    # 1448. Count Good Nodes in Binary Tree
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node: TreeNode, max_val: int):
            if not node:
                return 0
            good = 1 if node.val >= max_val else 0
            new_max_val = max(max_val, node.val)
            left = dfs(node.left, new_max_val)
            right = dfs(node.right, new_max_val)
            return left + right + good

        return dfs(root, root.val)
    # 437. Path Sum III
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        dic = defaultdict(int)
        dic[0] = 1

        def dfs(node: TreeNode, curr_sum: int):
            if not node:
                return 0
            new_curr_sum = curr_sum + node.val
            count = dic[new_curr_sum - targetSum]
            dic[new_curr_sum] += 1
            count += dfs(node.left, new_curr_sum)
            count += dfs(node.right, new_curr_sum)

            dic[new_curr_sum] -= 1

            return count

        return dfs(root, 0)


if __name__ == "__main__":
    # Create a binary tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(2)
    # Create a Solution instance and call the method
    solution = Solution()
    print(solution.maxLevelSum(root))  # Output: [1, 3, 4]
    print("1448: ", solution.goodNodes(root))  # 4
