using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Collections;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text.RegularExpressions;
using System.Text;
using System;

class Result
{

    /*
     * Complete the 'plusMinus' function below.
     *
     * The function accepts INTEGER_ARRAY arr as parameter.
     */

    public static void plusMinus(List<int> arr)
    {
        int posNum = 0;
        int negNum = 0;
        int zeroNum = 0;
        int arrLength = arr.Count;
        foreach (int digit in arr)
        {
            if (digit == 0)
            {
                zeroNum++;
            }
            else if (digit > 0)
            {
                posNum++;
            }
            else if (digit < 0)
            {
                negNum++;
            }
        }
        double zeroRate = (double)zeroNum / arrLength;
        double posRate = (double)posNum / arrLength;
        double negRate = (double)negNum / arrLength;
        System.Console.WriteLine(zeroRate.ToString("F6"));
        System.Console.WriteLine(negRate.ToString("F6"));
        System.Console.WriteLine(posRate.ToString("F6"));
    }

    /*
     * Complete the 'miniMaxSum' function below.
     *
     * The function accepts INTEGER_ARRAY arr as parameter.
     */

    public static void miniMaxSum(List<int> arr)
    {
        long minRes = 0;
        long maxRes = 0;
        long total = 0;
        long minNum = int.MaxValue;
        long maxNum = int.MinValue;
        // long maxNum = 0;
        foreach (long number in arr)
        {
            total += number;
            if (number < minNum)
            {
                minNum = number;
            }
            if (number > maxNum)
            {
                maxNum = number;
            }
        }
        maxRes = total - minNum;
        minRes = total - maxNum;

        System.Console.WriteLine(minRes.ToString() + " " + maxRes.ToString());
        // System.Console.WriteLine();
    }

    /*
     * Complete the 'timeConversion' function below.
     *
     * The function is expected to return a STRING.
     * The function accepts STRING s as parameter.
     */

    public static string timeConversion(string s)
    {
        string ampm = s.Substring(s.Length - 2);
        string[] splitTime = s.Substring(0, 8).Split(':');
        int hour = int.Parse(splitTime[0]);

        if (ampm == "PM")
        {
            if (hour != 12)
            {
                hour += 12;
            }
        }
        else
        {
            if (hour == 12) hour = 0;
        }
        return $"{hour:D2}:{splitTime[1]}:{splitTime[2]}";



    }
    /*
     * Complete the 'findMedian' function below.
     *
     * The function is expected to return an INTEGER.
     * The function accepts INTEGER_ARRAY arr as parameter.
     */
    public static int findMedian(List<int> arr)
    {
        arr = arr.Sort();
        int middleIndex = arr.Count / 2;
        int res = arr[middleIndex];
        return res;
    }

    /*
     * Complete the 'lonelyinteger' function below.
     *
     * The function is expected to return an INTEGER.
     * The function accepts INTEGER_ARRAY a as parameter.
     */

    public static int lonelyinteger(List<int> a)
    {
        Dictionary<int, int> map = new Dictionary<int, int>();
        foreach (int i in a)
        {
            if (map.ContainsKey(i))
            {
                map[i] += 1;
            }
            else
            {
                map[i] = 1;
            }
        }
        foreach (var pair in map)
        {
            if (pair.Value == 1)
            {
                return pair.Key;
            }
        }
        return 0;

    }

    /*
     * Complete the 'diagonalDifference' function below.
     *
     * The function is expected to return an INTEGER.
     * The function accepts 2D_INTEGER_ARRAY arr as parameter.
     */

    public static int diagonalDifference(List<List<int>> arr)
    {
        int arrLength = arr.Count;
        int diagSum = 0;
        int antiDiagSum = 0;
        for (int i = 0; i < arrLength; i++)
        {
            diagSum += arr[i][i];
            antiDiagSum += arr[i][arrLength - 1 - i];
        }
        return Math.Abs(diagSum - antiDiagSum);


    }

    /*
     * Complete the 'countingSort' function below.
     *
     * The function is expected to return an INTEGER_ARRAY.
     * The function accepts INTEGER_ARRAY arr as parameter.
     */

    public static List<int> countingSort(List<int> arr)
    {
        Dictionary<int, int> dic = new Dictionary<int, int>();
        foreach (int i in arr)
        {
            if (dic.ContainsKey(i))
            {
                dic[i] += 1;
            }
            else
            {
                dic[i] = 1;
            }
        }
        int[] res = new int[100];
        for (int i = 0; i < 100; i++)
        {
            if (dic.ContainsKey(i))
            {
                res[i] += dic[i];
            }
            else
            {
                res[i] = 0;
            }
        }
        return res.ToList();
    }
    public static int flippingMatrix(List<List<int>> matrix)
    {
        // answer is in python
    }

    /*
     * Complete the 'caesarCipher' function below.
     *
     * The function is expected to return a STRING.
     * The function accepts following parameters:
     *  1. STRING s
     *  2. INTEGER k
     */

    public static string caesarCipher(string s, int k)
    {
        k = k % 26;
        StringBuilder sb = new StringBuilder();
        foreach (char c in s)
        {
            if (char.IsLower(c))
            {
                char shifted = (char)('a' + (c - 'a' + k) % 26);
                sb.Append(shifted);
            }
            else if (char.IsUpper(c))
            {
                char shifted = (char)('A' + (c - 'A' + k) % 26);
                sb.Append(shifted);
            }
            else
            {
                sb.Append(c);
            }

        }
        return sb.ToString();

    }
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
            left--;
            right++;
        }
        return true;
    }
}

class Solution
{
    // dotnet new console -n MyApp：建立一次新的 C# 主控台應用程式專案，只需要在「第一次建立專案」的時候用到這行。
    //  下一次要編譯和執行，只做 cd MyApp => dotnet run

    public static void Main(string[] args)
    {
        // int n = Convert.ToInt32(Console.ReadLine().Trim());

        // List<int> arr = Console.ReadLine().TrimEnd().Split(' ').ToList().Select(arrTemp => Convert.ToInt32(arrTemp)).ToList();

        // Result.plusMinus(arr);
        Console.WriteLine(PalindromeIndex("bcbc"));
    }
}
