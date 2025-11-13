from collections import defaultdict, deque
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

    # 這是1137. N-th Tribonacci Number 的recursion寫法
    def tribonacci(self, n: int, memo={}) -> int:
        if n in memo:
            return memo[n]
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        memo[n] = (
            tribonacci(n - 1, memo) + tribonacci(n - 2, memo) + tribonacci(n - 3, memo)
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
