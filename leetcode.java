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
}
