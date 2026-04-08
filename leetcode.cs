

// create a simple C# class that i can do leetcode question here
using System;
using System.Collections.Generic;
using System.Text;

public class LeetCode
{
    public static void Main(string[] args)
    {
        Console.WriteLine("Hello, LeetCode!");
    }

    // You can add your LeetCode solutions here as methods.
    static int TwoSum(int[] nums, int target)
    {
        Dictionary<int, int> numDict = new Dictionary<int, int>();
        for (int i = 0; i < nums.Length; i++)
        {
            int complement = target - nums[i];
            if (numDict.ContainsKey(complement))
            {
                return new int[] { numDict[complement], i };
            }
            numDict[nums[i]] = i;
        }
        throw new ArgumentException("No two sum solution");
    }
    // 1768. Merge Strings Alternately
    static string MergeAlternately(string word1, string word2)
    {
        StringBuilder merged = new StringBuilder();
        int pointer1 = 0, pointer2 = 0;
        while (pointer1 < word1.Length || pointer2 < word2.Length)
        {
            if (pointer1 < word1.Length)
            {
                merged.Append(word1[pointer1]);
                pointer1++;
            }
            if (pointer2 < word2.Length)
            {
                merged.Append(word2[pointer2]);
                pointer2++;
            }
        }

        return merged.ToString();
    }
    
    // 1071. Greatest Common Divisor of Strings
    public string GcdOfStrings(string str1, string str2) {
        
    }
}   