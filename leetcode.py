class Solution:
    def lowestCommonAncestor(self, root, p, q):
        # 如果當前節點為 None，代表沒找到
        if not root:
            return None

        # 如果當前節點是 p 或 q，就回傳這個節點
        if root == p or root == q:
            return root

        # 往左右子樹遞迴尋找
        left = self.lowestCommonAncestor(root.left, p, q)   # 左子樹遞迴
        right = self.lowestCommonAncestor(root.right, p, q) # 右子樹遞迴

        # 如果左右子樹都找到了，說明當前節點是 LCA
        if left and right:
            return root

        # 否則回傳不為 None 的子樹結果
        return left if left else right
    