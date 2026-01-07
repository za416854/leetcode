from collections import defaultdict, deque, Counter
import math
import sys
from typing import List, Optional
import heapq


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
        dic = defaultdict(list)
        for a, b in connections:
            dic[a].append((b, 1))
            dic[b].append((a, 0))
        visited = set()

        def dfs(city: int):
            count = 0
            visited.add(city)
            for nei, found in dic[city]:
                if nei not in visited:
                    count += found + dfs(nei)
            return count

        return dfs(0)

    # 399. Evaluate Division
    def calcEquation(
        self, equations: List[List[str]], values: List[float], queries: List[List[str]]
    ) -> List[float]:
        graph = defaultdict(list)
        results = [float]
        for (a, b), val in zip(equations, values):
            graph[a].append((b, val))
            graph[b].append((a, 1 / val))

        def dfs(curr: str, target: str, acc: float, visited: set):
            if curr == target:
                return acc
            visited.add(curr)
            for nei, val in graph[curr]:
                if nei not in visited:
                    result = dfs(nei, target, acc * val, visited)
                    if result != -1:
                        return result
            return -1

        for a, b in queries:
            if a not in graph or b not in graph:
                results.append(-1)
            else:
                results.append(dfs(a, b, 1.0, set()))
        return results

    # 215. Kth Largest Element in an Array
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        return heap[0]

    # 2542. Maximum Subsequence Score
    def maxScore(self, nums1, nums2, k):
        pairs = sorted(
            zip(nums1, nums2), key=lambda x: -x[1]
        )  # 依照 index = 1 (nums2)由大到小排序
        curr = 0
        max_res = 0
        heap = []
        for num1, num2 in pairs:
            heapq.heappush(heap, num1)
            curr += num1
            if len(heap) > k:
                smallest = heapq.heappop(heap)
                curr -= smallest
            if len(heap) == k:
                max_res = max(max_res, curr * num2)
        return max_res

    # 2462. Total Cost to Hire K Workers
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        n = len(costs)
        left = 0
        right = n - 1
        left_heap, right_heap = [], []
        res = 0
        for _ in range(candidates):
            if left <= right:
                heapq.heappush(left_heap, costs[left])
                left += 1
            if left <= right:
                heapq.heappush(right_heap, costs[right])
                right -= 1
        for _ in range(k):
            if right_heap and (not left_heap or right_heap[0] < left_heap[0]):
                right_candidate = heapq.heappop(right_heap)
                res += right_candidate
                if left <= right:
                    heapq.heappush(right_heap, costs[right])
                    right -= 1
            else:
                left_candidate = heapq.heappop(left_heap)
                res += left_candidate
                if left <= right:
                    heapq.heappush(left_heap, costs[left])
                    left += 1
        return res

    # 374. Guess Number Higher or Lower
    def guessNumber(self, n: int) -> int:
        left = 1
        right = n
        while left <= right:
            mid = (left + right) // 2
            res = guess(mid)
            if res == 0:
                return mid
            elif res < 0:
                right = mid - 1
            else:
                left = mid + 1

    # 2300. Successful Pairs of Spells and Potions
    def successfulPairs(
        self, spells: List[int], potions: List[int], success: int
    ) -> List[int]:
        n = len(spells)
        m = len(potions)
        res = [0] * n
        potions.sort()
        for i in range(n):
            left = 0
            right = m - 1
            while left <= right:
                mid = left + (right - left) // 2
                product = spells[i] * potions[mid]
                if product >= success:
                    right = mid - 1
                else:
                    left = mid + 1
            res[i] = m - left
        return res

    # 162. Find Peak Element
    def findPeakElement(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left

    # 875. Koko Eating Bananas
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 這題思路是要用雙指針來找出最小的可以在h小時內吃完香蕉的根數，利用 can_finish去算 piles 裡每個 pile 要吃得時數，若<=h則繼續往左
        # 尋找有沒有更小可以在h小時內吃完香蕉的根數
        def can_finish(piles: List[int], h: int, k: int):
            total_hours = 0
            for pile in piles:
                total_hours += math.ceil(pile / k)
            return total_hours <= h

        left = 1
        right = max(piles)
        while left < right:
            mid = left + (right - left) // 2
            if can_finish(piles, h, mid):
                right = mid
            else:
                left = mid + 1
        return left

    # 17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        # 1️⃣ 數字到字母的映射表（模擬電話鍵盤）
        mapping = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        res = []  # 用來存最終結果

        # 2️⃣ 定義遞迴函數（DFS + 回朔）
        def backtrack(index, path):
            # Base case：如果處理完所有數字 → 收集結果
            if index == len(digits):
                res.append("".join(path))
                return

            # 取得當前數字對應的所有字母
            possible_letters = mapping[digits[index]]

            # 對每個可能字母進行遞迴
            for ch in possible_letters:
                # 譬如: path = ['a', 'e'] =>  res.append("ae") =>  path.pop()  # 回朔，變回 ['a']
                path.append(ch)  # ➕ 選擇（往下一層）
                backtrack(index + 1, path)  # 🔁 遞迴
                path.pop()  # ➖ 回朔（回上一層）

        # 3️⃣ 從第 0 個數字開始
        backtrack(0, [])

        return res

    # 216. Combination Sum III
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []

        def dfs(start: int, path: List, remain: int):
            if remain == 0 and len(path) == k:
                res.append(path[:])
                return
            if remain < 0 or len(path) > k:
                return
            for i in range(start, 10):
                path.append(i)
                dfs(i + 1, path, remain - i)
                path.pop()

        dfs(1, [], n)
        return res

    # 1137. N-th Tribonacci Number
    def tribonacci(self, n: int) -> int:
        # 🧩 Step 1: 處理基礎情況
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1

        # 🧠 Step 2: 初始化前三項 (T0, T1, T2)
        a, b, c = 0, 1, 1

        # 🚀 Step 3: 從第3項開始一路往上算到第n項
        for i in range(3, n + 1):
            a, b, c = b, c, a + b + c  # 同時更新三個值 (Tn-3, Tn-2, Tn-1 → Tn)

        # ✅ Step 4: 回傳最新的 c，也就是 Tn
        return c
        #  最後筆記: 所以DP就是比recur好的地方就是，他可以藉由儲存已經做過的事情記錄在變數裡面，以減少後續重複地計算的精神

    # DP - 1D
    # 這是1137. N-th Tribonacci Number 的recursion寫法
    def tribonacci(self, n: int, memo={}) -> int:
        if n in memo:
            return memo[n]
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        memo[n] = (
            self.tribonacci(n - 1, memo)
            + self.tribonacci(n - 2, memo)
            + self.tribonacci(n - 3, memo)
        )
        return memo[n]

    # 746. Min Cost Climbing Stairs
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        # bk track solution also brute-force solution
        n = len(cost)
        # def dfs(i):
        #     if i >= n:
        #         return 0
        #     one = cost[i] + dfs(i + 1)
        #     two = cost[i] + dfs(i + 2)
        #     return min(one, two)
        # return min(dfs(0), dfs(1))

        # DP solution
        # 思維其實不是到n -1 階所付出的cost，而是 第n階所付出的cost(會超出陣列index)。然後初始站在第0 跟1 階不算任何的費用，踏出去才算費用
        curr2, curr1 = 0, 0
        for i in range(2, len(cost) + 1):
            res = min(curr2 + cost[i - 2], curr1 + cost[i - 1])
            curr2, curr1 = curr1, res
        return curr1

    # 198. House Robber
    def rob(self, nums: List[int]) -> int:
        # bk track solution also brute-force solution
        # n = len(nums)
        # def dfs(i: int):
        #     if i >= n:
        #         return 0
        #     skip = dfs(i + 1)
        #     curr = nums[i] + dfs(i + 2)
        #     return max(skip, curr)
        # return dfs(0)
        curr1, curr2 = 0, 0
        for num in nums:
            res = max(curr1, curr2 + num)
            curr2, curr1 = curr1, res
        return curr1

    # 790. Domino and Tromino Tiling
    def numTilings(self, n: int) -> int:
        MOD = 10**9 + 7
        if n <= 2:
            return n
        if n == 3:
            return 5
        dp = [0] * (n + 1)
        dp[1], dp[2], dp[3] = 1, 2, 5
        for i in range(4, n + 1):
            # 這題非常難，若忘了要再去看推導的公式 https://leetcode.com/problems/domino-and-tromino-tiling/solutions/116581/detail-and-explanation-of-on-solution-wh-npb4/?envType=study-plan-v2&envId=leetcode-75
            # 才會對這邊得出來的該簡化公式的結果有較清楚的認識
            dp[i] = (2 * dp[i - 1] + dp[i - 3]) % MOD
        return dp[n]

    # DP - Multidimensional
    # 62. Unique Paths
    def uniquePaths(self, m: int, n: int) -> int:
        # 等價寫法
        # 1.
        # dp = [[0 for _ in range(n)] for _ in range(m)]
        # 2.
        # dp = []
        # for _ in range(m):
        #     row = []
        #     for _ in range(n):
        #         row.append(0)
        #     dp.append(row)
        # 創建一個 m x n 的 DP 表格
        dp = [[0] * n for _ in range(m)]

        # 1. KRIS:要先處理邊界條件 (第一行和第一列)
        # 如果在第一行或第一列，「選擇」是不是就只有一條路？
        # 第一行 (i=0) 只能從左邊到達，所以都是 1
        for j in range(n):
            dp[0][j] = 1

        # 第一列 (j=0) 只能從上方到達，所以都是 1
        for i in range(m):
            dp[i][0] = 1

        # 2. 應用遞歸關係
        for i in range(1, m):
            for j in range(1, n):
                # KRIS:這題用結果去想的話就是，結尾終點dp[m-1][n-1]一定是 來自上方 (dp[i-1][j])的所有可能 +上 來自左方 (dp[i][j-1])的所有可能
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        # 3. 返回右下角的最終結果
        return dp[m - 1][n - 1]

    # 1143. Longest Common Subsequence
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 暴力解: DFS + bk track
        # n1 = len(text1)
        # n2 = len(text2)
        # def dfs(i, j):
        #     if i == n1 and j == n2:
        #         return 0
        #     if text1[i] == text2[j]:
        #         return 1 + dfs(i + 1, j + 1)
        #     else:
        #         # 若不同 → 嘗試跳過任一方，取最大值
        #         skipA = dfs(i + 1, j)
        #         skipB = dfs(i, j + 1)
        #         return max(skipA, skipB)
        # return dfs(0, 0)

        # DP 1
        # memo = dict()
        # n1 = len(text1)
        # n2 = len(text2)

        # def dfs(i, j):
        #     if (i, j) in memo:
        #         return memo[(i, j)]
        #     if i == n1 or j == n2:
        #         return 0
        #     if text1[i] == text2[j]:
        #         memo[(i, j)] = 1 + dfs(i + 1, j + 1)
        #     else:
        #         memo[(i, j)] = max(dfs(i + 1, j), dfs(i, j + 1))
        #     return memo[(i, j)]

        # return dfs(0, 0)
        # DP 2 最推薦
        m = len(text1)
        n = len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    # max意義是: 若當前字母不一致，要取出上面位置前一個 跟左邊位置前一個 比較大的值放當前，繼續堆疊出結果(GPT範例圖片看到就一目了然，若忘了可以請AI產)
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    # 714. Best Time to Buy and Sell Stock with Transaction Fee
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # 這題核心不是要去保存買了之後怎樣然後往後面推算或是賣了之後怎樣往後推算，這比較偏貪心的思維
        # 這題是要用DP的思維下去思考，是要每天記錄賣/不賣，買/不買的結果，然後最後return 最優的cash(因為最後還是要賣掉得到最大獲利)
        cash, hold = 0, -prices[0]
        for price in prices[1:]:
            # 不賣 cash : 我昨天就沒持股，今天繼續保持沒持股- v.s. 賣掉股票hold + price - fee: 我昨天有持股，今天把股票賣掉
            cash = max(cash, hold + price - fee)
            # 不買 hold : 昨天就持股，今天繼續持股。 買入股票 cash - price: 昨天沒持股，今天花錢買股票
            hold = max(hold, cash - price)
        return cash

    # 72. Edit Distance
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        # 創建 (m+1) x (n+1) 的 DP 表格
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # 1. 初始化邊界條件 (Base Cases)
        # dp[i][0]：若word2為空字串，則word1需要刪除最多m次，以跟word2一致
        for i in range(m + 1):
            dp[i][0] = i
        # dp[0][j]：若word1為空字串，則word1需要插入最多n次，以跟word2一致
        for j in range(n + 1):
            dp[0][j] = j
        # 2. 填充表格
        for i in range(1, m + 1):
            for j in range(1, n + 1):

                # 比較當前字元 (注意索引 i-1, j-1)
                if word1[i - 1] == word2[j - 1]:
                    # 情況 A: 字元匹配，距離等於左上方 (不需操作)
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 情況 B: 字元不匹配，取三種操作的最小值 + 1
                    # 其實這題最難的是要想要怎麼思考出來把題目給的三個動作Replace, Delete, Insert變成二維的表格這件事，橫軸表示刪除i, 縱軸表示插入j，還有替換這個case(但因為替換就是dp[i-1][j-1] +1 所以是用對角線來表示)
                    # Delete 從上方來、Insert 從左方來、Replace 從左上方來。
                    dp[i][j] = 1 + min(
                        dp[i - 1][j - 1],  # 替換 (Replace)
                        dp[i - 1][j],  # 刪除 (Delete)
                        dp[i][j - 1],  # 插入 (Insert)
                    )
        # 3. 返回最終答案
        return dp[m][n]

    # Bit Manipulation
    # 338. Counting Bits
    def countBits(self, n: int) -> List[int]:
        # 這題還是DP的思維，因為偶數尾數永遠是0，奇數永遠是1，所以用這種退一位的dp方式慢慢得到越大數字的1的個數
        # 這題不用先給base case, 因為就是從dp[0] 慢慢開始往後面去做運算
        dp = [0] * (n + 1)

        # 從 i=1 開始循環到 n
        for i in range(1, n + 1):
            # 應用遞歸關係 (Bottom-Up 實現)
            # i >> 1: dp查詢先前已經有的 i/2 的結果(無條件捨去) (即 i 的二進位去掉最右邊一位)
            # i & 1: 檢查 i 的最右邊一位是否為 1(是否為odd number)
            dp[i] = dp[i >> 1] + (i & 1)

        return dp

    # 136. Single Number
    def singleNumber(self, nums: List[int]) -> int:
        # 字典統計頻率
        # dic = dict()
        # for num in nums:
        #     if num not in dic.keys():
        #         dic[num] = 1
        #     else:
        #         dic[num] += 1
        # res = 0
        # for k, v in dic.items():
        #     if v == 1:
        #         res = k
        # return res

        # 這個解法就是用到XOR( ^ 符號)概念，也就是兩個一樣的數的二進位，譬如說1101 and 1101 ，會互相抵銷變為0(可以看手寫筆記介紹XOR)，最後rturn 剩下來的數字就是了
        res = 0
        for num in nums:
            res ^= num
        return res

    # 1318. Minimum Flips to Make a OR b Equal to c
    def minFlips(self, a: int, b: int, c: int) -> int:
        # 這題要瞭解的是，a、b、c 在『同一個 bit 位置』上會有 8 種組合，因此用 a,b,c  & 1 拿到尾數二進位，再逐一比較，但因為這題有規定要讓 (a OR b) 等於 c ，所以
        flips = 0
        # c 有可能比a 或 b 小，所以一定要等到每個都為0才可以終止迴圈
        while a > 0 or b > 0 or c > 0:
            a_bit = a & 1
            b_bit = b & 1
            c_bit = c & 1
            # 第一種情況: c_bit == 0 表示 a_bit == 1 就要翻一次， b_bit == 1 也要翻一次(OR 代表要兩個都是0才是0)，所以就flips += a_bit + b_bit就很直觀
            if c_bit == 0:
                flips += a_bit + b_bit
            # c_bit == 1 的話，就是只要a_bit == 0 and b_bit == 0才會需要翻成1，其中一個為1就沒差繼續往下
            else:
                if a_bit == 0 and b_bit == 0:
                    flips += 1
            # 往右推一格
            a >>= 1
            b >>= 1
            c >>= 1

        return flips

    # Monotonic Stack
    # 739. Daily Temperatures
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []
        for i, temp in enumerate(temperatures):
            while stack and temp > temperatures[stack[-1]]:
                prev = stack.pop()
                res[prev] = i - prev
            stack.append(i)
        return res

    # 1268. Search Suggestions System
    def suggestedProducts(
        self, products: List[str], searchWord: str
    ) -> List[List[str]]:
        products.sort()
        trie = Trie()
        for p in products:
            trie.insert(p)

        res = []
        prefix = ""
        for ch in searchWord:
            prefix += ch
            suggestions = trie.searchPrefix(prefix)
            res.append(suggestions)
        return res

    # 1768. Merge Strings Alternately
    def mergeAlternately(self, word1: str, word2: str) -> str:
        res = ""
        i = 0
        j = 0
        while i < len(word1) or j < len(word2):
            if i < len(word1) and j < len(word2):
                res += word1[i]
                res += word2[j]
                i += 1
                j += 1
            elif i >= len(word1) and j < len(word2):
                res += word2[j]
                j += 1
            else:
                res += word1[i]
                i += 1
        return res

    # 151. Reverse Words in a String
    def reverseWords(self, s: str) -> str:
        words = s.split()
        # 這裡reverse return None所以不用放var在等號左邊
        words.reverse()
        return " ".join(words)

    # 238. Product of Array Except Self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)
        curr_L = 1
        for i in range(len(nums)):
            # j = i + 1
            res[i] *= curr_L
            curr_L = curr_L * nums[i]
        # for i in reversed(range(len(nums))):
        curr_R = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= curr_R
            curr_R = curr_R * nums[i]
        return res

    def increasingTriplet(self, nums: List[int]) -> bool:
        first = float("inf")
        second = float("inf")
        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True  # n > second
        return False

    # 443. String Compression 這題是in-space操作，不能開list額外空間，所以只能用two pointer的方式來操作chars空間，並回傳write指針代表長度
    def compress(self, chars: List[str]) -> int:
        length = len(chars)
        read = 0
        write = 0
        while read < length:
            char_start = read
            while read < length and chars[read] == chars[char_start]:
                read += 1

            count = read - char_start
            chars[write] = chars[char_start]
            write += 1
            if count > 1:
                for ch in str(count):
                    chars[write] = ch
                    write += 1
        return write

    # 283. Move Zeroes
    def moveZeroes(self, nums: List[int]) -> None:
        zero_counter = 0
        write = 0
        for num in nums:
            if num == 0:
                zero_counter += 1
            else:
                nums[write] = num
                write += 1
        for i in range(write, len(nums)):
            nums[i] = 0

    # 392. Is Subsequence
    def isSubsequence(self, s: str, t: str) -> bool:
        s_index = 0
        t_index = 0
        while s_index < len(s) and t_index < len(t):
            if s[s_index] == t[t_index]:
                s_index += 1
            t_index += 1
        return True if s_index == len(s) else False

    # 11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max_val = 0
        while left < right:
            width = right - left
            curr_val = 0
            if height[left] > height[right]:
                curr_val = height[right] * width
                right -= 1
            else:
                curr_val = height[left] * width
                left += 1
            if curr_val > max_val:
                max_val = curr_val
        return max_val
        # another solution
        left = 0
        right = len(height) - 1
        maxVolme = 0
        while left < right:
            currHeight = min(height[left], height[right])
            currWidth = right - left
            currVolume = currHeight * currWidth
            maxVolme = max(maxVolme, currVolume)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return maxVolme

    # 1679. Max Number of K-Sum Pairs 這題要記得排序
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        counter = 0
        left = 0
        right = len(nums) - 1
        while left < right:
            sum = nums[left] + nums[right]
            if sum == k:
                left += 1
                right -= 1
                counter += 1
            elif sum < k:
                left += 1
            else:
                right -= 1
        return counter

    # 643. Maximum Average Subarray I
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 該解法 TLE
        # if k == 1:
        #     return nums[0]
        # start = 0
        # end = start + k
        # length = len(nums)
        # max_num = 0
        # while end < length:
        #     curr_num = 0
        #     for i in range(start, end, 1):
        #         curr_num += nums[i]
        #     curr_num = curr_num / k
        #     if curr_num > max_num:
        #         max_num = curr_num
        #     start += 1
        #     end += 1
        # return max_num

        length = len(nums)
        curr_sum = sum(nums[:k])
        max_sum = curr_sum
        for i in range(k, length):
            left_index = i - k
            curr_sum -= nums[left_index]
            curr_sum += nums[k]
            if curr_sum > max_sum:
                max_sum = curr_sum
            # max_sum = max(max_sum, curr_sum)
        return max_sum / 4

    # 1456. Maximum Number of Vowels in a Substring of Given Length
    def maxVowels(self, s: str, k: int) -> int:
        vowels = ["a", "e", "i", "o", "u"]
        curr_counters = 0
        for i in range(k):
            if s[i] in vowels:
                curr_counters += 1
        max_counter = curr_counters
        for i in range(k, len(s)):
            left_index = i - k
            if s[left_index] in vowels:
                curr_counters -= 1
            if s[i] in vowels:
                curr_counters += 1
            max_counter = max(curr_counters, max_counter)
        return max_counter

    # 1004. Max Consecutive Ones III
    def longestOnes(self, nums: List[int], k: int) -> int:
        start = 0
        length = len(nums)
        max_lenth = 0
        zero_count = 0
        for end in range(length):
            if nums[end] == 0:
                zero_count += 1
            while zero_count > k:
                if nums[start] == 0:
                    zero_count -= 1
                start += 1
            curr_length = end - start + 1
            max_lenth = max(curr_length, max_lenth)
        return max_lenth

    # 1493. Longest Subarray of 1's After Deleting One Element
    def longestSubarray(self, nums: List[int]) -> int:
        start = 0
        length = len(nums)
        max_length = 0
        zero_counter = 0
        for end in range(length):
            if nums[end] == 0:
                zero_counter += 1
            while zero_counter > 1:
                if nums[start] == 0:
                    zero_counter -= 1
                start += 1
            curr_length = end - start
            max_length = max(curr_length, max_length)
        return max_length

    # 1732. Find the Highest Altitude
    def largestAltitude(self, gain: List[int]) -> int:
        res = [0] * (len(gain) + 1)
        for i in range(1, len(gain) + 1):
            res[i] = gain[i - 1] + res[i - 1]
        res_num = 0
        for num in res:
            res_num = max(res_num, num)
        return res_num
        # another way
        curr_altitude = 0
        max_altitude = 0
        for num in gain:
            curr_altitude += num
            max_altitude = max(max_altitude, curr_altitude)
        return max_altitude

    # 724. Find Pivot Index
    def pivotIndex(self, nums: List[int]) -> int:
        total_sum = 0
        for num in nums:
            total_sum += num
        left_sum = 0
        for i in range(len(nums)):
            right_sum = total_sum - left_sum - nums[i]
            if right_sum == left_sum:
                return i
            left_sum += nums[i]
        return -1

    # 2215. Find the Difference of Two Arrays
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        set1 = set(nums1)
        set2 = set(nums2)
        res1 = []
        for num in set1:
            if num not in set2:
                res1.append(num)
        res2 = []
        for num in set2:
            if num not in set1:
                res2.append(num)
        return [res1, res2]

    # 1207. Unique Number of Occurrences
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        dic = dict()
        for num in arr:
            if num not in dic:
                dic[num] = 1
            else:
                dic[num] += 1
        res = []
        for k, v in dic.items():
            if v in res:
                return False
            res.append(v)
        return True

    # 1657. Determine if Two Strings Are Close
    def closeStrings(self, word1: str, word2: str) -> bool:
        if len(word1) != len(word2):
            return False
        # collections.Counter 是 Python 標準庫中的一個類別，將一個列表或字串轉換成一個字典，其中鍵 (Key) 是元素，值 (Value) 是該元素出現的次數。
        # from collections import Counter
        # c = Counter("banana")
        # c 的結果是 {'b': 1, 'a': 3, 'n': 2}
        counter1 = Counter(word1)
        counter2 = Counter(word2)
        if set(counter1.keys()) != set(counter2.keys()):
            return False
        sorted1 = sorted(counter1.values())
        sorted2 = sorted(counter2.values())
        if sorted1 != sorted2:
            return False
        return True

    # 2352. Equal Row and Column Pairs
    def equalPairs(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dic = dict()
        # 這題counter也可以，效率也比較好，因為會自己算好
        counter = Counter()

        for row in grid:
            # dic[tuple(row)] = dic.get(tuple(row), 0) + 1
            counter[tuple(row)] += 1
        res = 0
        for i in range(m):
            col = []
            for j in range(n):
                col.append(grid[j][i])
            res += dic.get(tuple(col), 0)
        return res

    # 2390. Removing Stars From a String
    def removeStars(self, s: str) -> str:
        stack = list()
        for char in s:
            if char == "*" and stack:
                stack.pop()
                continue
            stack.append(char)

        return "".join(stack)

    # 735. Asteroid Collision
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for num in asteroids:
            while stack and num < 0 and stack[-1] > 0:
                if stack[-1] < abs(num):
                    stack.pop()
                    continue
                elif stack[-1] == abs(num):
                    stack.pop()
                    break
                else:
                    break
            # python 獨有 while else 介紹: else 會在 while「正常結束」時執行，如果你在 while 裡用 break，中斷 while，else 不會執行。
            else:
                stack.append(num)
        return stack

    # 394. Decode String
    def decodeString(self, s: str) -> str:
        stack = []
        curr_num = 0
        curr_str = ""
        for ch in s:
            if ch.isdigit():
                curr_num = curr_num * 10 + int(ch)
            elif ch == "[":
                stack.append((curr_str, curr_num))
                curr_num = 0
                curr_str = ""
            elif ch == "]":
                prev_str, prev_num = stack.pop()
                curr_str = prev_str + prev_num * curr_str
            else:
                curr_str += ch
        return curr_str


# Monotonic Stack
# 901. Online Stock Span
class StockSpanner:

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            curr_price, curr_span = self.stack.pop()
            span += curr_span
        self.stack.append((price, span))
        return span

    # 1268. Search Suggestions System
    def suggestedProducts(
        self, products: List[str], searchWord: str
    ) -> List[List[str]]:
        products.sort()
        trie = Trie()
        for p in products:
            trie.insert(p)

        res = []
        prefix = ""
        for ch in searchWord:
            prefix += ch
            suggestions = trie.searchPrefix(prefix)
            res.append(suggestions)
        return res

    # 1768. Merge Strings Alternately
    def mergeAlternately(self, word1: str, word2: str) -> str:
        res = ""
        i = 0
        j = 0
        while i < len(word1) or j < len(word2):
            if i < len(word1) and j < len(word2):
                res += word1[i]
                res += word2[j]
                i += 1
                j += 1
            elif i >= len(word1) and j < len(word2):
                res += word2[j]
                j += 1
            else:
                res += word1[i]
                i += 1
        return res

    # 151. Reverse Words in a String
    def reverseWords(self, s: str) -> str:
        words = s.split()
        # 這裡reverse return None所以不用放var在等號左邊
        words.reverse()
        return " ".join(words)

    # 238. Product of Array Except Self
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)
        curr_L = 1
        for i in range(len(nums)):
            # j = i + 1
            res[i] *= curr_L
            curr_L = curr_L * nums[i]
        # for i in reversed(range(len(nums))):
        curr_R = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= curr_R
            curr_R = curr_R * nums[i]
        return res

    def increasingTriplet(self, nums: List[int]) -> bool:
        first = float("inf")
        second = float("inf")
        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True  # n > second
        return False

    # 443. String Compression 這題是in-space操作，不能開list額外空間，所以只能用two pointer的方式來操作chars空間，並回傳write指針代表長度
    def compress(self, chars: List[str]) -> int:
        length = len(chars)
        read = 0
        write = 0
        while read < length:
            char_start = read
            while read < length and chars[read] == chars[char_start]:
                read += 1

            count = read - char_start
            chars[write] = chars[char_start]
            write += 1
            if count > 1:
                for ch in str(count):
                    chars[write] = ch
                    write += 1
        return write

    # 283. Move Zeroes
    def moveZeroes(self, nums: List[int]) -> None:
        zero_counter = 0
        write = 0
        for num in nums:
            if num == 0:
                zero_counter += 1
            else:
                nums[write] = num
                write += 1
        for i in range(write, len(nums)):
            nums[i] = 0

    # 392. Is Subsequence
    def isSubsequence(self, s: str, t: str) -> bool:
        s_index = 0
        t_index = 0
        while s_index < len(s) and t_index < len(t):
            if s[s_index] == t[t_index]:
                s_index += 1
            t_index += 1
        return True if s_index == len(s) else False

    # 11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        max_val = 0
        while left < right:
            width = right - left
            curr_val = 0
            if height[left] > height[right]:
                curr_val = height[right] * width
                right -= 1
            else:
                curr_val = height[left] * width
                left += 1
            if curr_val > max_val:
                max_val = curr_val
        return max_val
        # another solution
        left = 0
        right = len(height) - 1
        maxVolme = 0
        while left < right:
            currHeight = min(height[left], height[right])
            currWidth = right - left
            currVolume = currHeight * currWidth
            maxVolme = max(maxVolme, currVolume)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return maxVolme

    # 1679. Max Number of K-Sum Pairs 這題要記得排序
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        counter = 0
        left = 0
        right = len(nums) - 1
        while left < right:
            sum = nums[left] + nums[right]
            if sum == k:
                left += 1
                right -= 1
                counter += 1
            elif sum < k:
                left += 1
            else:
                right -= 1
        return counter

    # 643. Maximum Average Subarray I
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # 該解法 TLE
        # if k == 1:
        #     return nums[0]
        # start = 0
        # end = start + k
        # length = len(nums)
        # max_num = 0
        # while end < length:
        #     curr_num = 0
        #     for i in range(start, end, 1):
        #         curr_num += nums[i]
        #     curr_num = curr_num / k
        #     if curr_num > max_num:
        #         max_num = curr_num
        #     start += 1
        #     end += 1
        # return max_num

        length = len(nums)
        curr_sum = sum(nums[:k])
        max_sum = curr_sum
        for i in range(k, length):
            left_index = i - k
            curr_sum -= nums[left_index]
            curr_sum += nums[k]
            if curr_sum > max_sum:
                max_sum = curr_sum
            # max_sum = max(max_sum, curr_sum)
        return max_sum / 4

    # 1456. Maximum Number of Vowels in a Substring of Given Length
    def maxVowels(self, s: str, k: int) -> int:
        vowels = ["a", "e", "i", "o", "u"]
        curr_counters = 0
        for i in range(k):
            if s[i] in vowels:
                curr_counters += 1
        max_counter = curr_counters
        for i in range(k, len(s)):
            left_index = i - k
            if s[left_index] in vowels:
                curr_counters -= 1
            if s[i] in vowels:
                curr_counters += 1
            max_counter = max(curr_counters, max_counter)
        return max_counter

    # 1004. Max Consecutive Ones III
    def longestOnes(self, nums: List[int], k: int) -> int:
        start = 0
        length = len(nums)
        max_lenth = 0
        zero_count = 0
        for end in range(length):
            if nums[end] == 0:
                zero_count += 1
            while zero_count > k:
                if nums[start] == 0:
                    zero_count -= 1
                start += 1
            curr_length = end - start + 1
            max_lenth = max(curr_length, max_lenth)
        return max_lenth

    # 1493. Longest Subarray of 1's After Deleting One Element
    def longestSubarray(self, nums: List[int]) -> int:
        start = 0
        length = len(nums)
        max_length = 0
        zero_counter = 0
        for end in range(length):
            if nums[end] == 0:
                zero_counter += 1
            while zero_counter > 1:
                if nums[start] == 0:
                    zero_counter -= 1
                start += 1
            curr_length = end - start
            max_length = max(curr_length, max_length)
        return max_length

    # 1732. Find the Highest Altitude
    def largestAltitude(self, gain: List[int]) -> int:
        res = [0] * (len(gain) + 1)
        for i in range(1, len(gain) + 1):
            res[i] = gain[i - 1] + res[i - 1]
        res_num = 0
        for num in res:
            res_num = max(res_num, num)
        return res_num
        # another way
        curr_altitude = 0
        max_altitude = 0
        for num in gain:
            curr_altitude += num
            max_altitude = max(max_altitude, curr_altitude)
        return max_altitude

    # 724. Find Pivot Index
    def pivotIndex(self, nums: List[int]) -> int:
        total_sum = 0
        for num in nums:
            total_sum += num
        left_sum = 0
        for i in range(len(nums)):
            right_sum = total_sum - left_sum - nums[i]
            if right_sum == left_sum:
                return i
            left_sum += nums[i]
        return -1

    # 2215. Find the Difference of Two Arrays
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        set1 = set(nums1)
        set2 = set(nums2)
        res1 = []
        for num in set1:
            if num not in set2:
                res1.append(num)
        res2 = []
        for num in set2:
            if num not in set1:
                res2.append(num)
        return [res1, res2]

    # 1207. Unique Number of Occurrences
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        dic = dict()
        for num in arr:
            if num not in dic:
                dic[num] = 1
            else:
                dic[num] += 1
        res = []
        for k, v in dic.items():
            if v in res:
                return False
            res.append(v)
        return True

    # 1657. Determine if Two Strings Are Close
    def closeStrings(self, word1: str, word2: str) -> bool:
        if len(word1) != len(word2):
            return False
        # collections.Counter 是 Python 標準庫中的一個類別，將一個列表或字串轉換成一個字典，其中鍵 (Key) 是元素，值 (Value) 是該元素出現的次數。
        # from collections import Counter
        # c = Counter("banana")
        # c 的結果是 {'b': 1, 'a': 3, 'n': 2}
        counter1 = Counter(word1)
        counter2 = Counter(word2)
        if set(counter1.keys()) != set(counter2.keys()):
            return False
        sorted1 = sorted(counter1.values())
        sorted2 = sorted(counter2.values())
        if sorted1 != sorted2:
            return False
        return True

    # 2352. Equal Row and Column Pairs
    def equalPairs(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dic = dict()
        # 這題counter也可以，效率也比較好，因為會自己算好
        counter = Counter()

        for row in grid:
            # dic[tuple(row)] = dic.get(tuple(row), 0) + 1
            counter[tuple(row)] += 1
        res = 0
        for i in range(m):
            col = []
            for j in range(n):
                col.append(grid[j][i])
            res += dic.get(tuple(col), 0)
        return res

    # 2390. Removing Stars From a String
    def removeStars(self, s: str) -> str:
        stack = list()
        for char in s:
            if char == "*" and stack:
                stack.pop()
                continue
            stack.append(char)

        return "".join(stack)

    # 735. Asteroid Collision
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for num in asteroids:
            while stack and num < 0 and stack[-1] > 0:
                if stack[-1] < abs(num):
                    stack.pop()
                    continue
                elif stack[-1] == abs(num):
                    stack.pop()
                    break
                else:
                    break
            # python 獨有 while else 介紹: else 會在 while「正常結束」時執行，如果你在 while 裡用 break，中斷 while，else 不會執行。
            else:
                stack.append(num)
        return stack

    # 394. Decode String
    def decodeString(self, s: str) -> str:
        stack = []
        curr_num = 0
        curr_str = ""
        res = ""
        for ch in s:
            if ch.isdigit():
                curr_num = curr_num * 10 + int(ch)
            elif ch == "[":
                stack.append((curr_str, curr_num))
                curr_num = 0
                curr_str = ""
            elif ch == "]":
                prev_str, prev_num = stack.pop()
                # curr_str 代表「目前這一層」已經完整解好的，串，pop 之後要把這層結果塞回上一層 curr_str
                # 我原本的做法是用res 來儲存curr_str組好的結果，這會造成巢狀效果失效
                # 要稍微用recursion的觀念去想這邊
                # res += prev_str + prev_num * curr_str 這是錯的

                curr_str = prev_str + prev_num * curr_str
            else:
                curr_str += ch
        return curr_str

    # 933. Number of Recent Calls
    def predictPartyVictory(self, senate: str) -> str:
        r_queue = deque()
        d_queue = deque()
        lengh = len(senate)
        for i, ch in enumerate(senate):
            if ch == "R":
                r_queue.append(i)
            else:
                d_queue.append(i)
        while r_queue and d_queue:
            r_index = r_queue.popleft()
            d_index = d_queue.popleft()
            if r_index < d_index:
                r_queue.append(r_index + lengh)
            else:
                d_queue.append(d_index + lengh)
        return "Radiant" if r_queue else "Dire"

    # 649. Dota2 Senate
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = head
        slow = head
        fast = head
        if fast is None or fast.next is None:
            return None
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            prev = slow
            slow = slow.next
        prev.next = slow.next
        return head

    # 328. Odd Even Linked List
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        if len(head) == 1:
            return head
        odd = head
        even = head.next
        even_head = even
        while even is not None and even.next is not None:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = even_head
        return head

    # 206. Reverse Linked List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr is not None:
            curr_next = curr.next
            curr.next = prev
            prev = curr
            curr = curr_next
        return prev

    # 2130. Maximum Twin Sum of a Linked List
    def pairSum(self, head: Optional[ListNode]) -> int:
        slow = head
        fast = head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
        prev = None
        while slow is not None:
            slow_next = slow.next
            slow.next = prev
            prev = slow
            slow = slow_next
        first = head
        second = prev
        max_val = 0
        while second is not None:
            curr_val = first.val + second.val
            max_val = max(curr_val, max_val)
            first = first.next
            second = second.next
        return max_val

    # 435. Non-overlapping Intervals
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[1])
        kept_sub_arrs = 1
        compare_num = intervals[0][1]
        for i in range(1, len(intervals)):
            curr_num = intervals[i][0]
            if curr_num >= compare_num:
                kept_sub_arrs += 1
                compare_num = intervals[i][1]
            # compare_num = intervals[i][1]，放這邊不行，因為「貪心只更新『你選的』，不能更新『你丟掉的』」
        res = len(intervals) - kept_sub_arrs
        return res

    # 452. Minimum Number of Arrows to Burst Balloons
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        points.sort(key=lambda x: x[1])
        arrow_position = points[0][1]
        res = 1
        for start, end in points:
            if start > arrow_position:
                res += 1
                arrow_position = end
        return res

    # 104. Maximum Depth of Binary Tree
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        left = self.maxDepth(root.left) + 1
        right = self.maxDepth(root.right) + 1
        return max(left, right)

    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:

        leaves_1 = []
        leaves_2 = []

        def DFS(node: Optional[TreeNode], leaves: list):
            if not node:
                return
            if not node.left and not node.right:
                leaves.append(node.val)
            DFS(node.left, leaves)
            DFS(node.right, leaves)

        DFS(root1, leaves_1)
        DFS(root2, leaves_2)
        return leaves_2 == leaves_1

    # 1448. Count Good Nodes in Binary Tree
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node: TreeNode, max_so_far):
            if not node:
                return 0
            good = 1 if node.val >= max_so_far else 0
            max_val = max(node.val, max_so_far)
            left_good = dfs(node.left, max_val)
            right_good = dfs(node.right, max_val)
            return good + left_good + right_good

        return dfs(root, root.val)

    # 437. Path Sum III
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        dic = defaultdict(int)
        # if 樹只有一個節點，「沒有」放 dic[0] = 1，若明明有一條，但卻還是算到0條！
        dic[0] = 1

        def dfs(node: Optional[TreeNode], curr_sum: int) -> int:
            if not node:
                return 0
            # 1. 更新當前路徑總和
            curr_sum += node.val
            # 2. 檢查是否有任何「前綴」能讓我們減出 targetSum
            # 也就是尋找：curr_sum - targetSum
            count = dic.get(curr_sum - targetSum, 0)
            # 3. 把當前的 curr_sum 存入字典，供子節點使用
            dic[curr_sum] = dic.get(curr_sum, 0) + 1
            count += dfs(node.left, curr_sum)
            count += dfs(node.right, curr_sum)
            # 5. 【關鍵】回溯 (Backtracking)：
            # 當離開這個節點回到父節點時，要移除當前的前綴和
            # 避免不同支線的路徑互相干擾
            dic[curr_sum] -= 1
            return count

        return dfs(root, 0)

    # 1372. Longest ZigZag Path in a Binary Tree
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        self.max_path = 0

        def dfs(node: Optional[TreeNode], directions: str, curr_paths: int):
            if not node:
                return
            self.max_path = max(self.max_path, curr_paths)
            dfs(node.left, "left", curr_paths + 1 if directions == "right" else 1)
            dfs(node.right, "right", curr_paths + 1 if directions == "left" else 1)
            return self.max_path

        return dfs(root, "", self.max_path)

    # 236. Lowest Common Ancestor of a Binary Tree
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 若root是分散在左子樹跟右子樹，回傳祖先root
        if left and right:
            return root
        return left if left else right

    # 199. Binary Tree Right Side View
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level_len = len(queue)
            for i in range(level_len):
                node = queue.popleft()
                if i == level_len - 1:
                    res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res

    # 1161. Maximum Level Sum of a Binary Tree
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue = deque([root])
        max_num = float("-inf")
        counter = 0
        res = 0
        while queue:
            level_len = len(queue)
            counter += 1
            curr_val = 0
            for _ in range(level_len):
                node = queue.popleft()
                curr_val += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if curr_val > max_num:
                max_num = curr_val
                res = counter
        return res

    # 700. Search in a Binary Search Tree
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        def dfs(node: Optional[TreeNode]):
            if not node:
                return None
            if node.val == val:
                return node
            if node.val > val:
                return dfs(node.left)
            else:
                return dfs(node.right)

        return dfs(root)

    # 450. Delete Node in a BST 這題需要多次練習
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        def dfs(node: Optional[TreeNode], key: int):
            if not node:
                return None
            if node.val > key:
                node.left = dfs(node.left, key)
            elif node.val < key:
                node.right = dfs(node.right, key)
            else:
                if not node.right:
                    return node.left
                if not node.left:
                    return node.right
                curr = node.right
                while curr.left:
                    curr = curr.left
                node.val = curr.val
                node.right = dfs(node.right, curr.val)
            return node

        return dfs(root, key)

    # 841. Keys and Rooms 這題不難但要對dfs 是有graph 的想像因為這是 graph 的dfs 
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = set()

        def dfs(room: int):
            if room in visited:
                return
            visited.add(room)
            for key in rooms[room]:
                dfs(key)

        dfs(0)
        return len(visited) == len(rooms)

    # 547. Number of Provinces 
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        # dfs solution
        visited = set()
        provinces = 0
        n = len(isConnected)

        def dfs(city: int):
            for j in range(n):
                if j not in visited and isConnected[city][j] == 1:
                    visited.add(j)
                    dfs(j)

        for i in range(n):
            if i not in visited:
                visited.add(i)
                dfs(i)
                provinces += 1

        return provinces
        # bfs solution
        visited = set()
        provinces = 0
        n = len(isConnected)
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


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 933. Number of Recent Calls
class RecentCounter:

    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        self.queue.append(t)
        # self.queue.appendleft
        while self.queue and self.queue[0] < t - 3000:
            self.queue.popleft()
        return len(self.queue)


# 2336. Smallest Number in Infinite Set
class SmallestInfiniteSet:

    def __init__(self):
        self.current = 1  # 下一個尚未取出的自然數
        self.heap = []  # 被加回的數字
        self.added = set()  # 避免 heap 重複數字

    def popSmallest(self) -> int:
        if self.heap:  # 若有被加回的數，優先取最小
            smallest = heapq.heappop(self.heap)
            self.added.remove(smallest)
            return smallest
        else:
            val = self.current
            self.current += 1  # 往下一個自然數移動
            return val

    def addBack(self, num: int) -> None:
        if num < self.current and num not in self.added:
            heapq.heappush(self.heap, num)
            self.added.add(num)


# 208. Implement Trie (Prefix Tree)
class TrieNode2:

    def __init__(self):
        self.root = Trienode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trienode()
            node = node.children[ch]
        node.isWord = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.isWord

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True


# 1268. Search Suggestions System
class TrieNode2:
    def __init__(self):
        self.children = dict()
        self.suggestions = []


# 1268. Search Suggestions System
class Trie:
    def __init__(self):
        self.root = TrieNode2()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode2()

            node = node.children[ch]
            if len(node.suggestions) < 3:
                node.suggestions.append(word)

    def searchPrefix(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                return []
            node = node.children[ch]
        return node.suggestions


class Trienode:
    def __init__(self):
        self.children = dict()
        self.isWord = False


if __name__ == "__main__":
    # Create a binary tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(2)
    # Create a Solution instance and call the method
    solution = Solution()
    # print(solution.maxLevelSum(root))  # Output: [1, 3, 4]
    # print("1448: ", solution.goodNodes(root))  # 4
    # connections = [[0, 1], [1, 2], [2, 3], [3, 0]]
    # print(" ", solution.minReorder(4, connections))
    print(solution.decodeString("3[a2[c]]"))
