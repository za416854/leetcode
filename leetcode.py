import heapq


from typing import List


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


def find_busiest_time(data):
    pass
