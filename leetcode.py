import heapq
from collections import deque


from typing import List, Optional


class Solution:
    def lowestCommonAncestor(self, root, p, q):
        # 如果當前節點為 None，代表沒找到
        if not root:
            return None

        # 如果當前節點是 p 或 q，就回傳這個節點
        if root == p or root == q:
            return root

        # 往左右子樹遞迴尋找
        left = self.lowestCommonAncestor(root.left, p, q)  # 左子樹遞迴
        right = self.lowestCommonAncestor(root.right, p, q)  # 右子樹遞迴

        # 如果左右子樹都找到了，說明當前節點是 LCA
        if left and right:
            return root

        # 否則回傳不為 None 的子樹結果
        return left if left else right

    def sort_k_messed_array(arr, k):
        heap = []
        res = []
        # Step 1: 把前 k+1 個元素加入 heap
        for i in range(k + 1):
            heapq.heappush(heap, arr[i])
        # Step 2: 從第 k+1 個元素開始
        for i in range(k + 1, len(arr)):
            # 把 heap 中最小的取出（就是下一個排序應出現的值）
            res.append(heapq.heappop(heap))
            # 加入當前元素
            heapq.heappush(heap, arr[i])

        # Step 3: 把 heap 裡剩下的元素依序取出
        while heap:
            res.append(heapq.heappop(heap))
        return res


# heappush:👉 把一個元素 item 加進 heap 中，並維持最小堆的結構
# heap = []
# heapq.heappush(heap, 3)
# heapq.heappush(heap, 1)
# heapq.heappush(heap, 4)
# print(heap)  # 👉 輸出：[1, 3, 4]（最小值在前）

# heapq.heappop(heap)：👉 從 heap 中取出最小值，並移除它，維持堆的結構

# 為何要加一? 因為要放index = 0的第一個元素時，它最多只會在 index 0~2 出現（共 k+1 = 3 個位置）


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(self, head: ListNode) -> bool:
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if fast == slow:
            return True
    return False


# debug your code below
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)

# creates a linked list with a cycle: 1 -> 2 -> 3 -> 4 -> 2 (cycle)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node2

print(has_cycle(node1))


# Generate all combinations of well-formed parentheses. A parenthesis combination is well-formed if each opening parenthesis "(" is closed by a matching closing parenthesis ")" in the correct order.
def generate_parentheses(n: int) -> List[str]:
    if n == 0:
        return []
    res = []
    stack = []
    stack.append(("", 0, 0))
    while stack:
        current, open_count, close_count = stack.pop()
        if len(current) == 2 * n:
            res.append(current)
            continue
        if open_count < n:
            stack.append((current + "(", open_count + 1, close_count))
        if close_count < open_count:
            stack.append((current + ")", open_count, close_count + 1))
    return res

# Find the Duplicates
from typing import List

def find_duplicates(arr1: List[int], arr2: List[int]) -> List[int]:
    result = []
    i, j = 0, 0  # 雙指針初始化

    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            result.append(arr1[i])  # 元素相同 → 加入結果
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1  # 小的那個往前走
        else:
            j += 1

    return result

def isToeplitz(arr: List[List[int]]) -> bool:
    rows = len(matrix)
    cols = len(matrix[0])

    # Loop through each element (except last row and column)
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Compare current cell with bottom-right neighbor
            if matrix[i][j] != matrix[i + 1][j + 1]:
                return False  # Mismatch → not Toeplitz

    return True  # All diagonals matched

# 測試
print(find_duplicates([1, 2, 3, 5, 6], [2, 3, 4, 5]))  # [2, 3, 5]



def find_busiest_time(data):
    pass

# 236. Lowest Common Ancestor of a Binary Tree        
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root is None or root == p or q == root:
        return root
    # debug usage is quite useful for recursion, so we can set print() here to observe
    # print("Visiting:", root.val)
    # or 
    # print("At node", root.val, "left:", left, "right:", right) 
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    
    if left and right:
        return root
    
    return left if left else right
    

# for recursion practice 
# def recursion(self, node: TreeNode) -> TreeNode:
#     if BASE_CASE:
#         return node
#     left = recursion(node.left)
#     right = recursion(node.right)
    
#     return combine(left, right, node)

# 199. Binary Tree Right Side View 
'''
This one requires some imagination, we hv to spererate the tree into levels parellely and add the rightmost node val
The whole idea is: because level_size - 1 must be the rightmost one in each layer, 
so we need to append it. Then the for loop is to simulate the whole tree from left to right, 
and then slowly push it to the rightmost and append it.
'''
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    result = []
    queue = deque([root])  # Initialize BFS queue
    while queue:
        level_size = len(queue)  # Number of nodes at current level
        for i in range(level_size): # simulate that all nodes in this layer have been processed from leftmost to rightmost
            node = queue.popleft()
            # Add the last node of this level to result
            if i == level_size - 1: # represents this is the rightmost node of level
                result.append(node.val) # append it cause only seeable from right side 
            # Add children for the next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result

# 1161. Maximum Level Sum of a Binary Tree (BFS)
def maxLevelSum(self, root: Optional[TreeNode]) -> int:
    queue = deque([root])
    max_level = 1              # stores the level with the max sum
    level_number = 1          # tracks the current level number
    max_val = float('-inf')   # stores the maximum level sum seen so far
    while queue:
        level_size = len(queue)  # number of nodes at current level
        curr_val = 0             # sum of values at this level
        for _ in range(level_size):
            node = queue.popleft()
            curr_val += node.val
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if curr_val > max_val:
            max_val = curr_val
            max_level = level_number  # update max level correctly
        level_number += 1  # move to next level
    return max_level