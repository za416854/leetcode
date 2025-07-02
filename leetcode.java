import java.util.*;

public class KMessedSort {
    public static int[] sortKMessedArray(int[] arr, int k) {
        // 建立一個最小堆（min-heap）
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        
        int index = 0; // 用來放置排序後的元素

        // 將前 k+1 個元素放入 heap
        for (int i = 0; i <= k && i < arr.length; i++) {
            minHeap.add(arr[i]);
        }

        // 從第 k+1 個元素開始，處理整個陣列
        for (int i = k + 1; i < arr.length; i++) {
            arr[index++] = minHeap.poll(); // 取出 heap 中最小的元素放入結果
            minHeap.add(arr[i]);           // 將當前元素加入 heap
        }

        // 將剩下的 heap 中元素依序取出
        while (!minHeap.isEmpty()) {
            arr[index++] = minHeap.poll();
        }

        return arr; // 回傳已排序的陣列
    }
 
    public boolean hasCycle(ListNode head) {
        // 快慢指針初始化，都從頭節點開始
        ListNode slow = head;
        ListNode fast = head;

        // 當 fast 和 fast.next 都不為 null 時才繼續迴圈
        while (fast != null && fast.next != null) {
            slow = slow.next;         // 慢指針走一步
            fast = fast.next.next;    // 快指針走兩步

            // 如果快慢指針相遇，表示有環
            if (slow == fast) {
                return true;
            }
        }

        // 如果跳出迴圈，表示沒環（正常走完）
        return false;
    } 

}
