using System;

class Program
{

    public static int PalindromeIndex(string s)
    {
        int left = 0;
        int right = s.Length - 1;
        while (left < right)
        {
            if (s[left] != s[right])
            {
                if (IsPalindrome(s, left + 1, right))
                {
                    return left;
                }
                else if (IsPalindrome(s, left, right - 1))
                {
                    return right;
                }
                return -1;
                // 需要保留 return -1：

                // 例子：s = "abecbea"

                // 這個字串不是回文，我們來看看進入邏輯後會怎麼執行：

                // 一開始：

                // left = 0（s[0] = 'a'）

                // right = 6（s[6] = 'a'）
                // ✅ 相等 → left++、right--

                // 接著：

                // left = 1（s[1] = 'b'）

                // right = 5（s[5] = 'e'）
                // ❌ 不相等 → 進入 if (s[left] != s[right])

                //                     此時程式會檢查：

                // 移除左邊的 b，變成 "aecbea"，是否為回文？❌ 不是

                // 移除右邊的 e，變成 "abecba"，是否為回文？❌ 不是

                // 這代表不管刪左或刪右都無法變成回文！

                // 此時你就需要回傳 - 1，表示「無法變成回文」，也就是這行：

            }
            left++;
            right--;
        }
        return -1;
    }
    private static bool IsPalindrome(string s, int left, int right)
    {
        while (left < right)
        {
            if (s[left] != s[right])
            {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    // dotnet new console -n MyApp：建立一次新的 C# 主控台應用程式專案，只需要在「第一次建立專案」的時候用到這行。
    //  下一次要編譯和執行，只做 cd MyApp => dotnet run
    static void Main(string[] args)
    {
        // 這裡呼叫你原本在 HackerRank.cs 裡的函式
        Console.WriteLine(PalindromeIndex("bcbc")); // 0 or 3
        Console.WriteLine(PalindromeIndex("aaab")); // 3
        Console.WriteLine(PalindromeIndex("baa"));  // 0
        Console.WriteLine(PalindromeIndex("aaa"));  // -1 (已是回文)
    }
}
