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
}

class Solution
{
    public static void Main(string[] args)
    {
        int n = Convert.ToInt32(Console.ReadLine().Trim());

        List<int> arr = Console.ReadLine().TrimEnd().Split(' ').ToList().Select(arrTemp => Convert.ToInt32(arrTemp)).ToList();

        Result.plusMinus(arr);

    }
}
