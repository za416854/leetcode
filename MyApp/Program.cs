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
