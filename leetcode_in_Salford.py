from collections import defaultdict, deque
import sys
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

    # 1372. Longest ZigZag Path in a Binary Tree
    def longestZigZag(self, root: TreeNode) -> int:
        max_paths = 0

        def dfs(node: TreeNode, direction: str, curr_max_paths: int):
            nonlocal max_paths  # to let python know max_paths doent belong to dfs function
            if not node:
                return
            max_paths = max(max_paths, curr_max_paths)
            # direction == right doesnt affect that this line is going to left so dont be misled by 'left' or right string
            dfs(node.left, "left", curr_max_paths + 1 if direction == "right" else 1)
            dfs(node.right, "right", curr_max_paths + 1 if direction == "left" else 1)

        dfs(root, "", max_paths)
        return max_paths

    # 1. Two Sum, cause we gonna return the index of number so use dictionary to store {value,index}
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = dict()
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in dic.keys():
                return [dic[complement], i]
            dic[nums[i]] = i

    # 236. Lowest Common Ancestor of a Binary Tree
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if not root or root == p or root == q:
            return root

        res_left = self.lowestCommonAncestor(root.left, p, q)
        res_right = self.lowestCommonAncestor(root.right, p, q)

        if res_left and res_right:
            return root

        return res_left or res_right

    # 450. Delete Node in a BST
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left and not root.right:
                return None
            elif not root.left:
                return root.right
            elif not root.right:
                return root.left
            else:
                temp = root.right
                while temp.left:
                    temp = temp.left
                root.val = temp.val
                root.right = self.deleteNode(root.right, temp.val)
        return root

    # 700. Search in a Binary Search Tree
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None
        if val < root:
            root.left = self.searchBST(root.left, val)
        elif val > root:
            root.right = self.searchBST(root.right, val)
        else:
            return root

    # 230. Kth Smallest Element in a BST
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # first solution
        self.count = 0
        self.res = None

        def inorder(node: TreeNode):
            if not node:
                return None
            inorder(node.left)
            self.count += 1
            if self.count == k:
                self.res = node.val
            inorder(node.right)

        inorder(root)
        return self.res
        # second solution
        vals = list()

        def dfs(node: TreeNode):
            if not node:
                return
            dfs(node.left)
            vals.append(node.val)
            dfs(node.right)

        dfs(root)
        return vals[k - 1]

    # 530. Minimum Absolute Difference in BST
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        self.min_num = sys.maxsize
        self.prev = None

        def dfs(node: TreeNode):
            if not node:
                return None
            dfs(node.left)
            if self.prev is not None:
                self.min_num = min(self.min_num, node.val - self.prev)
            self.prev = node.val
            dfs(node.right)

        dfs(root)
        return self.min_num

    # 98. Validate Binary Search Tree
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        self.res = True
        self.prev = None

        def dfs(node: TreeNode):
            if not node:
                return
            dfs(node.left)
            if self.prev is not None:
                # The definition of BST is: for all nodes: the value of the left subtree is strictly less than the root node, and the value of the right subtree is strictly greater than the root node.
                if node.val <= self.prev:
                    self.res = False
            self.prev = node.val
            dfs(node.right)

        dfs(root)
        return self.res

    # 841. Keys and Rooms
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        # dfs solution
        visited = set()

        def dfs(room):
            if room in visited:
                return
            visited.add(room)
            for key in rooms[room]:
                dfs(key)

        dfs(0)
        return len(visited) == len(rooms)

        # bfs solution
        queue = deque([0])
        visited = set([0])
        while queue:
            room = queue.popleft()
            for key in rooms[room]:
                if key not in visited:
                    visited.add(key)
                    queue.append(key)
        return len(visited) == len(rooms)

    # 547. Number of Provinces
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        # dfs solution
        n = len(isConnected)
        visited = [False] * n
        provinces = 0

        def dfs(city):
            for j in range(n):
                if not visited[j] and isConnected[city][j] == 1:
                    visited[j] = True
                    dfs(j)

        for i in range(n):
            if not visited[i]:
                visited[i] = True
                dfs(i)
                provinces += 1
        return provinces

        # bfs solution
        n = len(isConnected)
        provinces = 0
        visited = set()
        for i in range(n):
            if i not in visited:
                queue = deque([i])
                while queue:
                    city = queue.popleft()
                    for j in range(n):
                        if j not in visited and isConnected[city][j] == 1:
                            visited.add(j)
                            queue.append(j)
                provinces += 1
        return provinces

    # 1466. Reorder Routes to Make All Paths Lead to the City Zero
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        graph = defaultdict(list)
        for a, b in connections:
            graph[a].append((b, 1))
            graph[b].append((a, 0))
        visited = set()
        def dfs(city):
            visited.add(city)
            changes = 0
            for nei, must_increase in graph[city]:
                if nei not in visited:
                    changes += must_increase + dfs(nei)
            return changes
        return dfs(0)
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
    connections = [[0, 1], [1, 2], [2, 3], [3, 0]]
    print(" ", solution.minReorder(4, connections))
