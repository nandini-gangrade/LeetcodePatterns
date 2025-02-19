## Table of Contents

| S. No | Topic | Link to Explanation | Link to LeetCode Discussion |
|-------|-----------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------|
| 1 | **Substring Problems (10-line Template)** | [Go to Explanation](#substring-problems) | [Link](https://leetcode.com/problems/minimum-window-substring/solutions/26808/Here-is-a-10-line-template-that-can-solve-most-substring'-problems/) |
| 2 | **Sliding Window Cheatsheet Template** | [Go to Explanation](#sliding-window) | [Link](https://leetcode.com/problems/frequency-of-the-most-frequent-element/solutions/1175088/C++Maximum-Sliding-Window-Cheatsheet-Template/) |
| 3 | **Two Pointers** | [Go to Explanation](#two-pointers) | [Link](https://leetcode.com/discuss/study-guide/1688903/Solved-all-two-pointers-problems-in-100-days) |
| 4 | **Backtracking** | [Go to Explanation](#backtracking) | [Link](https://medium.com/leetcode-patterns/leetcode-pattern-3-backtracking-5d9e5a03dc26) |
| 5 | **Backtracking Approach (Advanced)** | [Go to Explanation](#backtracking-approach) | [Link](https://leetcode.com/problems/permutations/solutions/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partitioning)/) |
| 6 | **Dynamic Programming Patterns** | [Go to Explanation](#dynamic-programming-patterns) | [Link](https://leetcode.com/discuss/study-guide/458695/Dynamic-Programming-Patterns) |
| 7 | **Dynamic Programming Patterns II** | [Go to Explanation](#dynamic-programming-patterns-ii) | [Link](https://leetcode.com/discuss/study-guide/1437879/Dynamic-Programming-Patterns) |
| 8 | **Binary Search Template** | [Go to Explanation](#binary-search) | [Link](https://leetcode.com/discuss/study-guide/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems) |
| 9 | **Graph (DFS & BFS Traversal)** | [Go to Explanation](#graph-dfs-bfs) | [Link](https://leetcode.com/discuss/study-guide/937307/Iterative-or-Recursive-DFS-and-BFS-Tree-Traversal-or-In-Pre-Post-and-LevelOrder) |
| 10 | **Graph for Beginners** | [Go to Explanation](#graph-for-beginners) | [Link](https://leetcode.com/discuss/study-guide/655708/Graph-For-Beginners-Problems-or-Pattern-or-Sample-Solutions) |
| 11 | **Monotonic Stack** | [Go to Explanation](#monotonic-stack) | [Link](https://leetcode.com/discuss/study-guide/2347639/A-comprehensive-guide-and-template-for-monotonic-stack-based-problems) |
| 12 | **Important String Questions Patterns** | [Go to Explanation](#string-patterns) | [Link](https://leetcode.com/discuss/study-guide/2001789/Collections-of-Important-String-questions-Pattern) |
| 13 | **Bit Manipulation & How to Use It** | [Go to Explanation](#bit-manipulation) | [Link](https://leetcode.com/discuss/interview-question/3695233/All-Types-of-Patterns-for-Bits-Manipulations-and-How-to-use-it) |
| 14 | **LeetCode Patterns (BFS, DFS, etc.)** | [Go to Explanation](#leetcode-patterns-bfs-dfs) | [Link](https://medium.com/leetcode-patterns/leetcode-pattern-1-bfs-dis-25-of-the-problems-part-1-51945038-4353) |
| 15 | **Greedy Approach for Interval-Based Problems** | [Go to Explanation](#greedy-interval) | [Link 1](https://leetcode.com/discuss/general-discussion/794725/General-Pattern-for-greedy-approachfor-Interval-based-problems) [Link 2](https://medium.com/setimpark0807/leetcode-is-easy-the-interval-pattern-d68a7c1c841) |
| 16 | **Prefix Sum & Difference Array** | [Go to Explanation](#prefix-sum) | [Link](https://leetcode.com/discuss/study-guide/1193713/Prefix-Sum-and-Difference-Array-CompleteStudy) |
| 17 | **Trie (Prefix Tree)** | [Go to Explanation](#trie-prefix-tree) | [Link](https://leetcode.com/discuss/study-guide/1129847/Ultimate-Guide-to-Trie) |

<br>

---

<br>

## 1. Substring Problems (10-line Template)
### 🔹 What is it?
Substring problems are a common category in string manipulation that require finding specific patterns or substrings within a given string.

### 🔹 When to Apply?
- Finding the smallest/largest substring satisfying a condition.
- Checking if a pattern exists in a string.
- Problems involving **sliding window** techniques.

### 🔹 General Template (Pseudocode)
```python
def sliding_window(s):
    left = 0
    for right in range(len(s)):
        # Expand the window by including s[right]

        while (window_condition_not_satisfied()):
            # Shrink the window from the left
            left += 1
```

### 🔹 Problem: **Minimum Window Substring**
#### 📝 Problem Statement:
Given two strings `s` and `t`, return the smallest substring in `s` that contains all characters of `t`.

#### 💡 Intuition:
We use the **Sliding Window** approach to keep track of the current window. We expand the window until all characters of `t` are included, then contract it to find the minimal valid window.

#### ✅ Approach:
1. Expand `right` pointer to include characters until all characters of `t` are in the window.
2. Contract `left` pointer to shrink the window while still containing all characters.
3. Track the minimum window size.

#### 🔹 Java Code:
```java
import java.util.HashMap;

class Solution {
    public String minWindow(String s, String t) {
        if (s.length() == 0 || t.length() == 0) return "";

        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : t.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        int left = 0, right = 0, minLen = Integer.MAX_VALUE, start = 0, count = map.size();
        while (right < s.length()) {
            char c = s.charAt(right);
            if (map.containsKey(c)) {
                map.put(c, map.get(c) - 1);
                if (map.get(c) == 0) count--;
            }
            right++;

            while (count == 0) {
                if (right - left < minLen) {
                    minLen = right - left;
                    start = left;
                }
                char leftChar = s.charAt(left);
                if (map.containsKey(leftChar)) {
                    map.put(leftChar, map.get(leftChar) + 1);
                    if (map.get(leftChar) > 0) count++;
                }
                left++;
            }
        }
        return minLen == Integer.MAX_VALUE ? "" : s.substring(start, start + minLen);
    }
}
```

#### 🔹 Python Code:
```python
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""

        count_t = Counter(t)
        window = {}
        left, right = 0, 0
        min_len, start, required = float("inf"), 0, len(count_t)

        while right < len(s):
            char = s[right]
            window[char] = window.get(char, 0) + 1
            if char in count_t and window[char] == count_t[char]:
                required -= 1
            right += 1

            while required == 0:
                if right - left < min_len:
                    min_len, start = right - left, left
                
                left_char = s[left]
                window[left_char] -= 1
                if left_char in count_t and window[left_char] < count_t[left_char]:
                    required += 1
                left += 1

        return "" if min_len == float("inf") else s[start:start + min_len]
```

#### 🔹 Related Problems:
1. **Longest Substring Without Repeating Characters** – (Expands and contracts window dynamically)
2. **Longest Repeating Character Replacement** – (Sliding window with a different constraint)
3. **Find All Anagrams in a String** – (Using window to match frequencies)

---

## 2. Sliding Window Cheatsheet Template

### 🔹 What is it?
Sliding Window is an optimized way to solve problems involving subarrays, substrings, or sequences by using a **moving window** instead of brute force.

### 🔹 When to Apply?
- Finding **maximum/minimum subarray sum**.
- Problems related to **consecutive elements**.
- **Count-based problems** (e.g., number of substrings with a given condition).

### 🔹 General Template (Pseudocode)
```python
left = 0  # Left boundary of window
for right in range(len(arr)):
    # Expand the window by including arr[right]
    
    while (condition_not_satisfied()):
        # Shrink window from the left
        left += 1
    
    # Update answer if needed
```

### 🔹 Problem: **Find the Length of the Longest Substring Without Repeating Characters**
#### 📝 Problem Statement:
Given a string `s`, find the length of the **longest substring** without repeating characters.

#### ✅ Approach:
1. Expand the **right pointer** until a duplicate character appears.
2. Shrink the **left pointer** until all characters are unique.
3. Keep track of the **maximum length**.

#### 🔹 Java Code:
```java
import java.util.HashSet;

class Solution {
    public int lengthOfLongestSubstring(String s) {
        HashSet<Character> set = new HashSet<>();
        int left = 0, maxLength = 0;
        
        for (int right = 0; right < s.length(); right++) {
            while (set.contains(s.charAt(right))) {
                set.remove(s.charAt(left));
                left++;
            }
            set.add(s.charAt(right));
            maxLength = Math.max(maxLength, right - left + 1);
        }
        return maxLength;
    }
}
```

#### 🔹 Python Code:
```python
def length_of_longest_substring(s: str) -> int:
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

#### 🔹 Related Problems:
1. **Minimum Window Substring**
2. **Longest Repeating Character Replacement**
3. **Find All Anagrams in a String**

---

## 3. Two Pointers
### 🔹 What is it?
Two Pointers technique is used when iterating through an array with two pointers moving at different speeds or in opposite directions.

### 🔹 When to Apply?
- Finding **pairs** in a sorted array.
- Problems involving **sorted arrays or linked lists**.
- **Merging two lists**.

### 🔹 General Template (Pseudocode)
```python
left, right = 0, len(arr) - 1  # Two pointers
while left < right:
    if (condition_met()):
        return result
    elif (need_to_move_left()):
        left += 1
    else:
        right -= 1
```

### 🔹 Problem: **Two Sum (Sorted Array)**
#### 📝 Problem Statement:
Given a **sorted** array `nums` and a target sum, return indices of two numbers such that they add up to the target.

#### ✅ Approach:
1. **Use two pointers** (`left` at start, `right` at end).
2. If `nums[left] + nums[right] > target`, move `right` **leftward**.
3. If `nums[left] + nums[right] < target`, move `left` **rightward**.
4. If equal, return indices.

#### 🔹 Java Code:
```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[]{left + 1, right + 1};
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
        return new int[]{}; // No solution
    }
}
```

#### 🔹 Python Code:
```python
def two_sum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        sum = nums[left] + nums[right]
        if sum == target:
            return [left + 1, right + 1]
        elif sum < target:
            left += 1
        else:
            right -= 1
    return []
```

#### 🔹 Related Problems:
1. **Container With Most Water**
2. **Trapping Rain Water**
3. **Merge Two Sorted Lists**

---

## 4. Backtracking  

**What is it?**  
Backtracking is a general algorithmic technique that tries to build a solution incrementally. If a partial solution fails, it backtracks and tries a different path. This is widely used for problems involving permutations, combinations, and constraint satisfaction problems like Sudoku, N-Queens, and Subset Sum.  

**When to Apply?**  
- When the problem requires all possible solutions (e.g., permutations, combinations).  
- When the problem has constraints, and you need to explore multiple possibilities efficiently.  
- When DFS-style recursion is a natural fit.  

**General Template (Pseudocode):**  
```python
def backtrack(path, choices):
    if base_case:  # Check if a solution is found
        save_solution()
        return
    for choice in choices:
        make_choice(choice)
        backtrack(path + choice, updated_choices)
        undo_choice(choice)  # Backtrack step
```

### **Problem: Generate All Subsets**  
**Question:**  
Given an array of distinct integers `nums`, return all possible subsets (the power set).  

**Example:**  
```
Input: nums = [1,2,3]
Output: [[],[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]
```

**Intuition & Approach:**  
- This is a classic backtracking problem where we explore every subset.  
- Start with an empty subset and try adding each element while recursively exploring further subsets.  
- Use recursion and backtracking to explore both **including** and **excluding** each number.  

#### **Java Code:**
```java
import java.util.*;

class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums, 0);
        return result;
    }

    private void backtrack(List<List<Integer>> result, List<Integer> tempList, int[] nums, int start) {
        result.add(new ArrayList<>(tempList)); // Add current subset to result

        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]); // Choose element
            backtrack(result, tempList, nums, i + 1); // Recurse
            tempList.remove(tempList.size() - 1); // Backtrack (remove last added element)
        }
    }
}
```

#### **Python Code:**
```python
class Solution:
    def subsets(self, nums):
        result = []
        def backtrack(start, path):
            result.append(path[:])  # Append a copy of path to result
            for i in range(start, len(nums)):
                path.append(nums[i])  # Include nums[i]
                backtrack(i + 1, path)  # Recurse with next index
                path.pop()  # Backtrack by removing last added element
        
        backtrack(0, [])
        return result
```

### **Related Problems:**  
1. **Combination Sum** ([Link](https://leetcode.com/problems/combination-sum/))  
   - Uses backtracking to explore all combinations of numbers that sum up to a target.  

2. **Permutations** ([Link](https://leetcode.com/problems/permutations/))  
   - Generates all possible orderings of elements using backtracking.  

3. **N-Queens** ([Link](https://leetcode.com/problems/n-queens/))  
   - Places queens on a chessboard using backtracking to avoid conflicts.  

4. **Word Search** ([Link](https://leetcode.com/problems/word-search/))  
   - Uses DFS with backtracking to search for words in a grid.  

---

## **5. Backtracking Approach (Advanced)**  

#### **What is Backtracking?**  
Backtracking is a **recursive** algorithm used for solving problems where you explore all possible solutions and "backtrack" when a certain condition fails. It is commonly used for **permutations, combinations, and constraint-satisfaction problems**.  

#### **When to Use?**  
- **Subset generation** (Power set, Combination Sum, etc.)  
- **Permutations** (Rearranging elements in all possible ways)  
- **Solving puzzles** (Sudoku, N-Queens, Word Search)  
- **Pathfinding problems** (Rat in a maze, Knight’s tour)  

#### **General Template for Backtracking**
```python
def backtrack(path, choices):
    if end_condition(path):  # Base case: A valid solution is found
        result.append(path[:])  # Store a copy
        return
    
    for choice in choices:
        path.append(choice)  # Make a choice
        backtrack(path, choices)  # Recur with the choice made
        path.pop()  # Undo the choice (Backtrack)
```

### **Problem: Generate All Permutations of an Array**  
**Question:**  
Given an array of distinct integers, return all possible permutations.  

**Example:**  
```
Input: nums = [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

#### **Approach:**  
1. Use a **recursive backtracking function** to generate all permutations.  
2. At each step, try adding each number that hasn’t been used yet.  
3. If a valid permutation is found (length == original array), add it to the result.  
4. Undo the last choice (backtrack) to explore other possibilities.  

#### **Java Solution:**  
```java
import java.util.*;

class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums);
        return result;
    }

    private void backtrack(List<List<Integer>> result, List<Integer> tempList, int[] nums) {
        if (tempList.size() == nums.length) {
            result.add(new ArrayList<>(tempList));  // Store a copy of tempList
            return;
        }

        for (int num : nums) {
            if (tempList.contains(num)) continue;  // Skip duplicates
            tempList.add(num);
            backtrack(result, tempList, nums);
            tempList.remove(tempList.size() - 1);  // Backtrack
        }
    }
}
```

#### **Python Solution:**  
```python
class Solution:
    def permute(self, nums):
        result = []
        def backtrack(path):
            if len(path) == len(nums):
                result.append(path[:])  # Add a copy of path
                return
            
            for num in nums:
                if num not in path:  # Avoid duplicates
                    path.append(num)
                    backtrack(path)
                    path.pop()  # Backtrack

        backtrack([])
        return result
```

---

### **Related Problems:**  
1. **Subsets (Power Set)** - [LeetCode Problem](https://leetcode.com/problems/subsets/)  
2. **Combination Sum** - [LeetCode Problem](https://leetcode.com/problems/combination-sum/)  
3. **N-Queens Problem** - [LeetCode Problem](https://leetcode.com/problems/n-queens/)  
4. **Sudoku Solver** - [LeetCode Problem](https://leetcode.com/problems/sudoku-solver/)  


🔗 **Reference for Backtracking Patterns:** [Click Here](https://leetcode.com/problems/permutations/solutions/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partitioning)/)  

---

## 6. Dynamic Programming Patterns  

**What is it?**  
Dynamic Programming (DP) is an optimization technique used to solve problems by breaking them into smaller overlapping subproblems and storing solutions to avoid redundant calculations. It is useful when problems exhibit **optimal substructure** and **overlapping subproblems**.  

**When to Apply?**  
- When a problem has recursive overlapping subproblems.  
- When the problem follows the **optimal substructure** property (solution of a bigger problem depends on its subproblems).  
- When brute-force recursion leads to redundant calculations.  

**General Template (Pseudocode):**  

**Top-Down (Memoization)**  
```python
def dp(i, state):
    if base_case: return result
    if memo[i][state] is not None: return memo[i][state]
    memo[i][state] = compute(dp(subproblems))
    return memo[i][state]
```

**Bottom-Up (Tabulation)**  
```python
dp = [[0] * cols for _ in range(rows)]
for i in range(rows):
    for j in range(cols):
        dp[i][j] = compute(dp[i-1][j], dp[i][j-1])
return dp[final_state]
```

### **Problem: Fibonacci Number**  
**Question:**  
Find the `n`th Fibonacci number where `F(0) = 0`, `F(1) = 1`, and `F(n) = F(n-1) + F(n-2)`.  

**Example:**  
```
Input: n = 5  
Output: 5  
Explanation: Fibonacci sequence: [0, 1, 1, 2, 3, 5]
```

**Intuition & Approach:**  
- We solve this using **Top-Down (Memoization)** and **Bottom-Up (Tabulation)** DP.  
- **Top-Down:** Use recursion with memoization to store intermediate results.  
- **Bottom-Up:** Build up results iteratively, avoiding recursion overhead.  

#### **Java Code:**
```java
import java.util.*;

class Solution {
    private Map<Integer, Integer> memo = new HashMap<>();

    public int fib(int n) {
        if (n <= 1) return n;
        if (memo.containsKey(n)) return memo.get(n);

        int result = fib(n - 1) + fib(n - 2);
        memo.put(n, result);
        return result;
    }
}
```

#### **Python Code:**
```python
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        memo = [-1] * (n + 1)
        def dp(n):
            if n <= 1:
                return n
            if memo[n] != -1:
                return memo[n]
            memo[n] = dp(n - 1) + dp(n - 2)
            return memo[n]
        return dp(n)
```

### **Related Problems:**  
1. **Climbing Stairs** ([Link](https://leetcode.com/problems/climbing-stairs/))  
   - Uses DP to count the number of ways to reach the top.  

2. **Longest Increasing Subsequence** ([Link](https://leetcode.com/problems/longest-increasing-subsequence/))  
   - Uses DP to find the longest increasing subsequence in an array.  

3. **0/1 Knapsack Problem** ([Link](https://leetcode.com/problems/last-stone-weight-ii/))  
   - Uses DP to maximize value while staying within weight constraints.  

4. **Coin Change** ([Link](https://leetcode.com/problems/coin-change/))  
   - Uses DP to compute the minimum number of coins needed to reach a target amount.  

---

## 7. **Dynamic Programming Patterns II**  

#### **What is it?**  
This section expands on DP by exploring more advanced techniques like **bitmask DP, state compression, Kadane’s Algorithm, DP with bitwise operations, and DP on trees/graphs**.  

#### **When to Apply?**  
- When DP needs additional **state representation** (like bitmasks).  
- When handling **multi-dimensional DP** problems.  
- When **optimizing DP with additional techniques** like Kadane’s Algorithm.  

#### **General Template (Pseudocode)**  

**Bitmask DP (Used in Traveling Salesman Problem, etc.)**  
```python
def dp(mask, pos):
    if all_visited(mask):
        return base_case
    if memo[mask][pos] is not None:
        return memo[mask][pos]
    result = compute(dp(mask | (1 << new_pos), new_pos))
    memo[mask][pos] = result
    return result
```

**Kadane’s Algorithm (Used in Maximum Subarray Sum)**  
```python
max_so_far = float('-inf')
max_ending_here = 0
for i in range(n):
    max_ending_here = max(arr[i], max_ending_here + arr[i])
    max_so_far = max(max_so_far, max_ending_here)
return max_so_far
```

### **Problem: Maximum Subarray (Kadane’s Algorithm)**  
**Question:**  
Find the contiguous subarray within an array that has the largest sum.  

**Example:**  
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]  
Output: 6  
Explanation: The subarray [4,-1,2,1] has the maximum sum.
```

**Intuition & Approach:**  
- Instead of checking all subarrays (`O(n^2)`), we use Kadane’s Algorithm (`O(n)`).  
- Maintain `max_ending_here` (current subarray sum) and `max_so_far` (global max).  
- If `max_ending_here` becomes negative, reset it to 0.  

#### **Java Code:**
```java
class Solution {
    public int maxSubArray(int[] nums) {
        int maxSoFar = Integer.MIN_VALUE, maxEndingHere = 0;

        for (int num : nums) {
            maxEndingHere += num;
            if (maxSoFar < maxEndingHere) {
                maxSoFar = maxEndingHere;
            }
            if (maxEndingHere < 0) {
                maxEndingHere = 0;
            }
        }
        return maxSoFar;
    }
}
```

#### **Python Code:**
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_so_far = float('-inf')
        max_ending_here = 0

        for num in nums:
            max_ending_here += num
            max_so_far = max(max_so_far, max_ending_here)
            if max_ending_here < 0:
                max_ending_here = 0

        return max_so_far
```

### **Related Problems:**  
1. **Best Time to Buy and Sell Stock** ([Link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/))  
   - Uses DP to track min price and max profit dynamically.  

2. **Partition Equal Subset Sum** ([Link](https://leetcode.com/problems/partition-equal-subset-sum/))  
   - Uses DP to check if an array can be partitioned into two subsets of equal sum.  

3. **Jump Game II** ([Link](https://leetcode.com/problems/jump-game-ii/))  
   - Uses DP/greedy approach to minimize jumps needed to reach the end of an array.  

4. **Travelling Salesman Problem (TSP)** (Advanced DP)  
   - Uses **bitmask DP** to minimize total travel distance.  

---

## 8. **Binary Search Template**  

#### **What is it?**  
Binary Search is a divide-and-conquer algorithm that finds an element in a sorted list by repeatedly dividing the search space in half.  

#### **When to Apply?**  
- When **array/list is sorted** or can be sorted easily.  
- When the problem asks for **minimum/maximum value** that satisfies a condition.  
- When the problem involves **searching in a monotonic function** (i.e., increasing or decreasing order).  

#### **General Template (Pseudocode)**  
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2  # Middle element
        if arr[mid] == target:
            return mid  # Found target
        elif arr[mid] < target:
            left = mid + 1  # Move right
        else:
            right = mid - 1  # Move left
    return -1  # Not found
```

### **Problem: Search in Rotated Sorted Array**  
**Question:**  
Given a rotated sorted array, find the index of a target value in `O(log n)`.  

**Example:**  
```
Input: nums = [4,5,6,7,0,1,2], target = 0  
Output: 4  
```

**Intuition & Approach:**  
1. Find which half of the array is sorted.  
2. If the target lies in the sorted half, apply **binary search** there.  
3. Otherwise, apply **binary search** on the unsorted half.  

#### **Java Code:**  
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) return mid;

            // Determine which half is sorted
            if (nums[left] <= nums[mid]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;  // Search left half
                } else {
                    left = mid + 1;   // Search right half
                }
            } else {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;   // Search right half
                } else {
                    right = mid - 1;  // Search left half
                }
            }
        }
        return -1;  // Not found
    }
}
```

#### **Python Code:**  
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid

            if nums[left] <= nums[mid]:  # Left half is sorted
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1  # Not found
```


### **Related Problems:**  
1. **Find First and Last Position of Element in Sorted Array** ([Link](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/))  
   - Uses binary search to find the leftmost and rightmost occurrences of a number.  

2. **Median of Two Sorted Arrays** ([Link](https://leetcode.com/problems/median-of-two-sorted-arrays/))  
   - Uses binary search on the smaller array for efficient median calculation.  

3. **Find Peak Element** ([Link](https://leetcode.com/problems/find-peak-element/))  
   - Uses binary search to find a peak element in an array.  

4. **Search a 2D Matrix** ([Link](https://leetcode.com/problems/search-a-2d-matrix/))  
   - Uses binary search to find an element in a matrix by treating it as a 1D array.  

---

## 9. **Graph (DFS & BFS)**  

#### **What is it?**  
Graphs are a collection of nodes (vertices) connected by edges. **DFS (Depth-First Search)** and **BFS (Breadth-First Search)** are two fundamental traversal techniques used to explore graphs.  

- **DFS**: Goes deep into a path before backtracking (stack-based).  
- **BFS**: Explores all neighbors first before moving to the next level (queue-based).  

#### **When to Apply?**  
- When you need to **traverse or search** in a graph or tree.  
- When solving **shortest path problems** (BFS is useful for unweighted graphs).  
- When dealing with **connected components**, **cycles**, and **topological sorting**.  

#### **General Template (DFS - Recursive & Iterative)**  
```python
# Recursive DFS
def dfs_recursive(graph, node, visited):
    if node in visited:
        return
    visited.add(node)
    for neighbor in graph[node]:
        dfs_recursive(graph, neighbor, visited)

# Iterative DFS
def dfs_iterative(graph, start):
    stack, visited = [start], set()
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node])  # Add neighbors
```

#### **General Template (BFS - Iterative)**  
```python
from collections import deque

def bfs(graph, start):
    queue, visited = deque([start]), set()
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node])  # Add neighbors
```


### **Problem: Number of Islands (Using BFS & DFS)**  
**Question:**  
Given a `grid` of `'1'` (land) and `'0'` (water), count the number of islands. An island is formed by adjacent `1`s (vertically/horizontally).  

**Example:**  
```
Input:
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

**Intuition & Approach:**  
1. Traverse the grid.  
2. When a `'1'` (land) is found, initiate DFS/BFS to mark all connected parts as visited.  
3. Count each DFS/BFS call as an island.  

#### **Java Code (DFS Approach):**  
```java
class Solution {
    public void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == '0')
            return;

        grid[i][j] = '0'; // Mark visited
        dfs(grid, i + 1, j); // Down
        dfs(grid, i - 1, j); // Up
        dfs(grid, i, j + 1); // Right
        dfs(grid, i, j - 1); // Left
    }

    public int numIslands(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(grid, i, j);
                }
            }
        }
        return count;
    }
}
```

#### **Python Code (BFS Approach):**  
```python
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid: return 0
        rows, cols = len(grid), len(grid[0])
        visited = set()
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        def bfs(r, c):
            queue = deque([(r, c)])
            while queue:
                row, col = queue.popleft()
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1' and (nr, nc) not in visited:
                        queue.append((nr, nc))
                        visited.add((nr, nc))
            grid[r][c] = '0'  # Mark visited
        
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    count += 1
                    visited.add((r, c))
                    bfs(r, c)
        return count
```


### **Related Problems:**  
1. **Word Search (Backtracking & DFS)** ([Link](https://leetcode.com/problems/word-search/))  
   - Uses DFS to explore word paths in a 2D grid.  

2. **Clone Graph** ([Link](https://leetcode.com/problems/clone-graph/))  
   - Uses DFS or BFS to copy an entire graph.  

3. **Surrounded Regions** ([Link](https://leetcode.com/problems/surrounded-regions/))  
   - Uses BFS/DFS to modify surrounded regions in a 2D board.  

4. **Course Schedule (Topological Sort - BFS/DFS)** ([Link](https://leetcode.com/problems/course-schedule/))  
   - Uses graph traversal to determine if all courses can be completed.  

---

## **10. Graph for Beginners** 🚀  

#### **What is a Graph?**  
A **graph** is a data structure that consists of **nodes (vertices)** connected by **edges**. It can be:  
- **Directed** (edges have direction) or **Undirected**  
- **Weighted** (edges have weights) or **Unweighted**  
- **Cyclic** (contains cycles) or **Acyclic**  

#### **When to Use Graphs?**  
- **Shortest path problems** (Dijkstra’s Algorithm, Bellman-Ford)  
- **Graph traversal** (DFS, BFS)  
- **Cycle detection**  
- **Topological sorting**  
- **Connected components detection**  


### **General Graph Traversal Algorithms**  

#### **1. Depth-First Search (DFS) - Recursive**
```python
def dfs(graph, node, visited):
    if node in visited:
        return
    visited.add(node)
    for neighbor in graph[node]:
        dfs(graph, neighbor, visited)
```

#### **2. Breadth-First Search (BFS) - Iterative**
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node])
```


### **Problem: Number of Connected Components in an Undirected Graph**  
**Question:**  
Given `n` nodes and a list of edges, return the number of **connected components** in the graph.  

**Example:**  
```
Input: n = 5, edges = [[0, 1], [1, 2], [3, 4]]
Output: 2
Explanation: There are two connected components: {0,1,2} and {3,4}.
```

#### **Approach:**  
1. Build an **adjacency list** from the given edges.  
2. Use **DFS or BFS** to traverse each component.  
3. Count the number of **times a DFS/BFS starts**, as it represents a new component.  


### **Java Solution (Using DFS)**
```java
import java.util.*;

class Solution {
    public int countComponents(int n, int[][] edges) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]); // Undirected graph
        }

        boolean[] visited = new boolean[n];
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs(graph, i, visited);
                count++; // New component found
            }
        }
        return count;
    }

    private void dfs(List<List<Integer>> graph, int node, boolean[] visited) {
        if (visited[node]) return;
        visited[node] = true;
        for (int neighbor : graph.get(node)) {
            dfs(graph, neighbor, visited);
        }
    }
}
```


### **Python Solution (Using BFS)**
```python
from collections import deque

class Solution:
    def countComponents(self, n, edges):
        graph = {i: [] for i in range(n)}
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)  # Undirected graph
        
        visited = set()
        count = 0
        
        def bfs(node):
            queue = deque([node])
            while queue:
                curr = queue.popleft()
                for neighbor in graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        for node in range(n):
            if node not in visited:
                visited.add(node)
                bfs(node)
                count += 1  # New component found
        
        return count
```

### **Related Problems:**  
1. **Graph Valid Tree** - [LeetCode Problem](https://leetcode.com/problems/graph-valid-tree/)  
   - Checks if a graph forms a valid tree (no cycles and connected).  
2. **Find if Path Exists in Graph** - [LeetCode Problem](https://leetcode.com/problems/find-if-path-exists-in-graph/)  
   - Uses DFS/BFS to check if there's a path between two nodes.  
3. **Course Schedule (Topological Sorting)** - [LeetCode Problem](https://leetcode.com/problems/course-schedule/)  
   - Uses **DFS or BFS** to check if all courses can be completed.  
4. **Rotting Oranges (BFS on Grid)** - [LeetCode Problem](https://leetcode.com/problems/rotting-oranges/)  
   - Uses BFS to find the minimum time to rot all oranges in a grid.  


🔗 **Reference for Graphs:** [Click Here](https://leetcode.com/discuss/study-guide/655708/Graph-For-Beginners-Problems-or-Pattern-or-Sample-Solutions)  

---

## **11. Monotonic Stack** 🔥  

#### **What is a Monotonic Stack?**  
A **Monotonic Stack** is a **special type of stack** where elements are maintained in **either increasing or decreasing order**. It is useful for **range-based problems** like **Next Greater Element** or **Stock Span Problem**.

- **Monotonic Increasing Stack**: Elements are pushed in **increasing order** (smallest at the bottom).  
- **Monotonic Decreasing Stack**: Elements are pushed in **decreasing order** (largest at the bottom).  

#### **When to Use?**  
- **Next Greater/Smaller Element**  
- **Stock Span Problem**  
- **Histogram Largest Rectangle Area**  
- **Trapping Rain Water**  


### **Problem: Next Greater Element**
**Question:**  
Given an array `nums`, find the **next greater element** for each element in the array. If no greater element exists, return `-1` for that index.

**Example:**  
```
Input: nums = [2, 1, 2, 4, 3]
Output: [4, 2, 4, -1, -1]
```


### **Approach:**
1. Use a **Monotonic Decreasing Stack**.  
2. Traverse the array **backwards**.  
3. For each element, **pop** from the stack until we find a greater element.  
4. Push the current element onto the stack.  


### **Java Solution**
```java
import java.util.*;

class Solution {
    public int[] nextGreaterElement(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();
        
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() <= nums[i]) {
                stack.pop(); // Remove smaller elements
            }
            result[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i]); // Push current element
        }
        
        return result;
    }
}
```

### **Python Solution**
```python
class Solution:
    def nextGreaterElement(self, nums):
        stack = []
        result = [-1] * len(nums)

        for i in range(len(nums) - 1, -1, -1):
            while stack and stack[-1] <= nums[i]:
                stack.pop()  # Remove smaller elements
            if stack:
                result[i] = stack[-1]
            stack.append(nums[i])  # Push current element
        
        return result
```


### **Related Problems:**  
1. **Next Greater Element I, II, III** - [LeetCode](https://leetcode.com/problems/next-greater-element-i/)  
2. **Largest Rectangle in Histogram** - [LeetCode](https://leetcode.com/problems/largest-rectangle-in-histogram/)  
3. **Daily Temperatures** - [LeetCode](https://leetcode.com/problems/daily-temperatures/)  
4. **Trapping Rain Water** - [LeetCode](https://leetcode.com/problems/trapping-rain-water/)  


🔗 **Reference for Monotonic Stack:** [Click Here](https://leetcode.com/discuss/general-discussion/528013/monotonic-stack-explained-using-visuals)  

---

## **12. Important String Questions Patterns** 🔠  

### **What are String Patterns?**  
String pattern problems involve techniques used to **search, modify, or analyze** strings efficiently. Some common techniques include:  
- **Sliding Window** (for substring problems)  
- **Two Pointers** (for palindrome checking, etc.)  
- **KMP Algorithm** (for pattern matching)  
- **Trie Data Structure** (for prefix-based search)  

### **When to Use?**  
- **Substring search problems**  
- **Finding longest palindromic substring**  
- **Pattern matching in strings**  
- **String compression or transformation problems**  

### **General Template (Pseudocode)**
#### **Expand Around Center Approach for Longest Palindromic Substring**
```plaintext
1. Define function longestPalindrome(s)
    a. If s is empty, return ""
    b. Initialize start and end pointers to track longest palindrome

2. Iterate through each character in string s
    a. Compute palindrome length with character as center (odd length)
    b. Compute palindrome length with adjacent characters as center (even length)
    c. Choose the longer palindrome and update start and end pointers

3. Define function expandAroundCenter(left, right)
    a. Expand outwards while left and right characters match
    b. Return length of the palindrome found

4. Return substring from start to end
```


### **Problem: Longest Palindromic Substring**  
**Question:**  
Given a string `s`, return the **longest palindromic substring** in `s`.

**Example:**  
```
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
```


#### **Java Solution**
```java
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
        int start = 0, end = 0;

        // Loop through each character as center
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandFromCenter(s, i, i);     // Odd length palindrome
            int len2 = expandFromCenter(s, i, i + 1); // Even length palindrome
            int len = Math.max(len1, len2);
            
            // Update the start and end pointers if we found a longer palindrome
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    // Helper function to expand around center
    private int expandFromCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;  // Move left pointer leftwards
            right++; // Move right pointer rightwards
        }
        return right - left - 1; // Length of palindrome
    }
}
```

#### **Python Solution**
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""
        start, end = 0, 0
        
        # Helper function to expand around center
        def expandAroundCenter(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1  # Move left pointer leftwards
                right += 1  # Move right pointer rightwards
            return right - left - 1  # Return palindrome length
        
        for i in range(len(s)):
            len1 = expandAroundCenter(i, i)  # Odd length palindrome
            len2 = expandAroundCenter(i, i + 1)  # Even length palindrome
            max_len = max(len1, len2)
            
            # Update start and end pointers
            if max_len > end - start:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
                
        return s[start:end + 1]
```


### **Related Problems:**  
| Problem | Explanation | Link |
|---------|------------|------|
| **Longest Palindromic Substring** | Find the longest substring that is a palindrome | [LeetCode](https://leetcode.com/problems/longest-palindromic-substring/) |
| **Repeated Substring Pattern** | Check if a string is formed by repeating a substring | [LeetCode](https://leetcode.com/problems/repeated-substring-pattern/) |
| **Find All Anagrams in a String** | Find all anagrams of a word in a given string | [LeetCode](https://leetcode.com/problems/find-all-anagrams-in-a-string/) |
| **Group Anagrams** | Group words that are anagrams of each other | [LeetCode](https://leetcode.com/problems/group-anagrams/) |


🔗 **Reference for String Patterns:** [Click Here](https://leetcode.com/discuss/study-guide/2001789/Collections-of-Important-String-questions-Pattern)  

---

## **13. Bit Manipulation Patterns** ⚡  

### **What is Bit Manipulation?**  
Bit manipulation involves **directly operating on binary representations** of numbers. It is used for:  
- **Efficient computation** (e.g., checking even/odd, multiplication/division by powers of 2)  
- **Optimizing space complexity** (e.g., bitwise operations in sets)  
- **Low-level programming** (e.g., system programming, cryptography)  


### **Common Bitwise Operators**  
| Operator | Symbol | Usage |
|----------|--------|--------|
| AND | `&` | `a & b` (sets bits to 1 if both are 1) |
| OR  | `|` | `a | b` (sets bits to 1 if either is 1) |
| XOR | `^` | `a ^ b` (sets bits to 1 if they are different) |
| NOT | `~` | `~a` (inverts bits) |
| Left Shift | `<<` | `a << n` (multiplies by `2^n`) |
| Right Shift | `>>` | `a >> n` (divides by `2^n`) |


## **When to Use?**  
- **Finding unique numbers in an array**  
- **Counting set bits in a number**  
- **Checking if a number is a power of two**  
- **Efficient swapping without extra space**  


### **General Template (Pseudocode)**
#### **Find Single Non-Duplicate in an Array (XOR Property)**
```plaintext
1. Define function findUnique(arr)
2. Initialize result = 0
3. Iterate through array:
    a. Perform result = result XOR element
4. Return result (unique number)
```


### **Problem: Find the Single Non-Duplicate Number**
**Question:**  
Given a list of integers where every number appears twice **except one**, find that single number.

**Example:**  
```
Input: nums = [4,1,2,1,2]
Output: 4
Explanation: 4 appears only once.
```


#### **Java Solution**
```java
class Solution {
    public int singleNumber(int[] nums) {
        int result = 0;
        
        // XOR all numbers, duplicates cancel out
        for (int num : nums) {
            result ^= num; 
        }
        
        return result; // Remaining number is unique
    }
}
```

#### **Python Solution**
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        
        # XOR all elements, duplicates cancel out
        for num in nums:
            result ^= num
            
        return result
```

### **Other Bitwise Tricks**
| Problem | Explanation | Link |
|---------|------------|------|
| **Power of Two** | Check if a number is a power of two using `n & (n-1) == 0` | [LeetCode](https://leetcode.com/problems/power-of-two/) |
| **Counting Set Bits** | Count 1s in binary representation | [LeetCode](https://leetcode.com/problems/number-of-1-bits/) |
| **Reverse Bits** | Reverse the bits of a number | [LeetCode](https://leetcode.com/problems/reverse-bits/) |


🔗 **Reference for Bit Manipulation Patterns:** [Click Here](https://leetcode.com/discuss/study-guide/1768823/Bit-Manipulation-Patterns-You-Must-Know)  

---

## **14. LeetCode Patterns (BFS, DFS, etc.)** 🚀  

### **What is BFS and DFS?**  
Breadth-First Search (BFS) and Depth-First Search (DFS) are two fundamental graph traversal techniques:  

- **BFS (Breadth-First Search)** explores neighbors first before going deeper.  
- **DFS (Depth-First Search)** goes deep into one branch before backtracking.  

### **When to Use?**  
- **BFS** is used for shortest path problems (e.g., shortest path in an unweighted graph, level-order traversal of a tree).  
- **DFS** is useful for exploring all possibilities (e.g., backtracking, cycle detection).  

### **General Template (Pseudocode)**  

#### **BFS Template (Using Queue)**
```plaintext
1. Initialize an empty queue and a set for visited nodes
2. Add the starting node to the queue
3. While the queue is not empty:
   a. Dequeue the front node
   b. Process the node
   c. Add all unvisited neighbors to the queue
```

#### **DFS Template (Using Recursion)**
```plaintext
1. Define a recursive function dfs(node)
2. If node is visited, return
3. Mark node as visited
4. Process the node
5. Recursively visit all neighbors
```

### **Problem: Shortest Path in an Unweighted Graph**  
**Question:** Given a graph represented as an adjacency list, find the shortest path from node `0` to all other nodes.  

**Example:**  
```
Input: edges = [[0,1],[0,2],[1,2],[1,3],[2,3],[3,4]], start = 0
Output: [0, 1, 1, 2, 3]
Explanation: The shortest distance from node 0 to all others.
```

### **Java Solution (BFS)**
```java
import java.util.*;

class Solution {
    public int[] shortestPath(int n, int[][] edges, int start) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) graph.add(new ArrayList<>());
        
        // Build adjacency list
        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        
        int[] distance = new int[n];
        Arrays.fill(distance, -1); // Mark unvisited nodes
        Queue<Integer> queue = new LinkedList<>();
        
        queue.offer(start);
        distance[start] = 0;

        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbor : graph.get(node)) {
                if (distance[neighbor] == -1) {
                    distance[neighbor] = distance[node] + 1;
                    queue.offer(neighbor);
                }
            }
        }
        
        return distance;
    }
}
```

### **Python Solution (BFS)**
```python
from collections import deque

class Solution:
    def shortestPath(self, n: int, edges: List[List[int]], start: int) -> List[int]:
        graph = {i: [] for i in range(n)}
        
        # Build adjacency list
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        distance = [-1] * n  # Mark unvisited nodes
        queue = deque([start])
        distance[start] = 0

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if distance[neighbor] == -1:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        
        return distance
```

### **Other BFS/DFS Problems**  
| Problem | Explanation | Link |
|---------|------------|------|
| **Number of Islands** | Use DFS/BFS to count connected components in a grid | [LeetCode](https://leetcode.com/problems/number-of-islands/) |
| **Word Ladder** | BFS for shortest transformation sequence | [LeetCode](https://leetcode.com/problems/word-ladder/) |
| **Clone Graph** | Use DFS/BFS to copy a graph | [LeetCode](https://leetcode.com/problems/clone-graph/) |

🔗 **Reference for BFS/DFS Patterns:** [Click Here](https://medium.com/leetcode-patterns/leetcode-pattern-1-bfs-dis-25-of-the-problems-part-1-51945038-4353)  

---

## **15. Greedy Approach for Interval-Based Problems 🔥**  

### **What is the Greedy Approach?**  
A greedy algorithm makes the best choice at each step, assuming this leads to the optimal solution. It works well in problems involving **intervals, scheduling, and resource allocation**.  

### **Common Interval-Based Problems**  
- **Activity Selection Problem** (Choosing maximum non-overlapping intervals)  
- **Meeting Rooms Problem** (Checking overlapping intervals)  
- **Merging Intervals**  
- **Minimum Platforms** (Railway station scheduling)  

### **Greedy Algorithm Template (Pseudocode)**
```plaintext
1. Sort the given intervals by their start or end time
2. Initialize an empty result list (or a counter)
3. Iterate through the sorted intervals:
   a. If the interval does not overlap, add it to the result
   b. If overlapping, take the optimal choice (based on the problem statement)
4. Return the result (count, merged intervals, etc.)
```

### **Problem: Maximum Number of Non-Overlapping Intervals**  
**Question:**  
You are given `n` intervals. Find the maximum number of non-overlapping intervals that can be selected.  

**Example:**  
```
Input: [[1,3], [2,4], [3,5], [7,9]]
Output: 3
Explanation: We can select intervals [1,3], [3,5], and [7,9].
```

#### **Java Solution**
```java
import java.util.*;

class Solution {
    public int maxNonOverlappingIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        
        // Step 1: Sort intervals by end time
        Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
        
        int count = 0;
        int end = Integer.MIN_VALUE;
        
        // Step 2: Select intervals greedily
        for (int[] interval : intervals) {
            if (interval[0] >= end) { // No overlap
                count++;
                end = interval[1]; // Update end time
            }
        }
        
        return count;
    }
}
```

#### **Python Solution**
```python
class Solution:
    def maxNonOverlappingIntervals(self, intervals):
        if not intervals:
            return 0

        # Step 1: Sort intervals by their end time
        intervals.sort(key=lambda x: x[1])
        
        count = 0
        end = float('-inf')
        
        # Step 2: Select intervals greedily
        for start, finish in intervals:
            if start >= end:  # No overlap
                count += 1
                end = finish  # Update end time
        
        return count
```

### **Other Greedy Interval Problems**  
| Problem | Explanation | Link |
|---------|------------|------|
| **Merging Overlapping Intervals** | Merge intervals if they overlap | [LeetCode](https://leetcode.com/problems/merge-intervals/) |
| **Meeting Rooms II** | Find the minimum meeting rooms required | [LeetCode](https://leetcode.com/problems/meeting-rooms-ii/) |
| **Minimum Number of Platforms** | Solve train scheduling problem | [GeeksForGeeks](https://www.geeksforgeeks.org/minimum-number-platforms-required-railwaybus-station/) |

🔗 **Reference for Greedy Interval Problems:** [Click Here](https://leetcode.com/discuss/study-guide/1935075/All-Important-Greedy-Problems-and-Patterns)

---

## **16. Prefix Sum & Difference Array**  

### **What is Prefix Sum?**  
Prefix sum is a technique used to preprocess an array so that range sum queries can be answered efficiently in **O(1) time** instead of recalculating the sum repeatedly.  

**Formula:**  
If `prefixSum[i]` stores the sum from `arr[0]` to `arr[i]`:  
\[
prefixSum[i] = prefixSum[i-1] + arr[i]
\]


### **Prefix Sum Algorithm (Pseudocode)**  
```plaintext
1. Initialize prefixSum[0] = arr[0]
2. Iterate through the array:
   prefixSum[i] = prefixSum[i-1] + arr[i]
3. To get sum of range [L, R]:  
   sum = prefixSum[R] - prefixSum[L-1]
```

#### **Example**
```
Input: arr = [2, 4, 1, 7, 3]
Prefix Sum: [2, 6, 7, 14, 17]
Query Sum(1,3) = PrefixSum[3] - PrefixSum[0] = 14 - 2 = 12
```

### **Java Implementation**
```java
class PrefixSum {
    // Function to calculate prefix sum array
    public static int[] computePrefixSum(int[] arr) {
        int n = arr.length;
        int[] prefixSum = new int[n];
        prefixSum[0] = arr[0];
        
        // Compute prefix sum
        for (int i = 1; i < n; i++) {
            prefixSum[i] = prefixSum[i - 1] + arr[i];
        }
        
        return prefixSum;
    }
    
    // Function to get range sum
    public static int getRangeSum(int[] prefixSum, int L, int R) {
        if (L == 0) return prefixSum[R];
        return prefixSum[R] - prefixSum[L - 1];
    }

    public static void main(String[] args) {
        int[] arr = {2, 4, 1, 7, 3};
        int[] prefixSum = computePrefixSum(arr);
        
        System.out.println("Sum of range (1,3): " + getRangeSum(prefixSum, 1, 3));
    }
}
```


### **Python Implementation**
```python
class PrefixSum:
    def __init__(self, arr):
        self.prefix_sum = [0] * len(arr)
        self.prefix_sum[0] = arr[0]
        
        # Compute prefix sum
        for i in range(1, len(arr)):
            self.prefix_sum[i] = self.prefix_sum[i - 1] + arr[i]

    def get_range_sum(self, L, R):
        if L == 0:
            return self.prefix_sum[R]
        return self.prefix_sum[R] - self.prefix_sum[L - 1]

# Example usage
arr = [2, 4, 1, 7, 3]
ps = PrefixSum(arr)
print("Sum of range (1,3):", ps.get_range_sum(1, 3))
```


### **What is Difference Array?**
Difference array is used when multiple range updates are required efficiently.

#### **Difference Array Algorithm (Pseudocode)**  
```plaintext
1. Create a difference array `diff[]` of size (n+1), initialized to 0.
2. To increment range [L, R] by `val`:
   diff[L] += val
   diff[R+1] -= val
3. Compute prefix sum of `diff[]` to get the final updated array.
```

#### **Example**
```
Input: arr = [0, 0, 0, 0, 0]
Update (1,3) by +5
Difference Array: [0, 5, 0, 0, -5]
Final Array: [5, 5, 5, 0, 0]
```

#### **Java Implementation of Difference Array**
```java
class DifferenceArray {
    public static int[] applyRangeUpdates(int[] arr, int[][] updates) {
        int n = arr.length;
        int[] diff = new int[n + 1];

        // Apply difference array technique
        for (int[] update : updates) {
            int L = update[0], R = update[1], val = update[2];
            diff[L] += val;
            if (R + 1 < n) diff[R + 1] -= val;
        }

        // Compute prefix sum to get the final updated array
        arr[0] = diff[0];
        for (int i = 1; i < n; i++) {
            arr[i] = arr[i - 1] + diff[i];
        }
        
        return arr;
    }

    public static void main(String[] args) {
        int[] arr = new int[5];
        int[][] updates = {{1, 3, 5}};
        
        int[] result = applyRangeUpdates(arr, updates);
        System.out.println(Arrays.toString(result)); // Output: [5, 5, 5, 0, 0]
    }
}
```

#### **Python Implementation of Difference Array**
```python
class DifferenceArray:
    def __init__(self, size):
        self.diff = [0] * (size + 1)

    def update(self, L, R, val):
        self.diff[L] += val
        if R + 1 < len(self.diff):
            self.diff[R + 1] -= val

    def get_final_array(self):
        arr = [0] * (len(self.diff) - 1)
        arr[0] = self.diff[0]
        for i in range(1, len(arr)):
            arr[i] = arr[i - 1] + self.diff[i]
        return arr

# Example usage
arr_size = 5
diff_array = DifferenceArray(arr_size)
diff_array.update(1, 3, 5)
print(diff_array.get_final_array())  # Output: [5, 5, 5, 0, 0]
```


### **Applications of Prefix Sum & Difference Array**
| Problem | Explanation | Link |
|---------|------------|------|
| **Range Sum Query** | Quickly compute sum of subarrays | [LeetCode](https://leetcode.com/problems/range-sum-query-immutable/) |
| **Range Addition** | Apply multiple range updates efficiently | [LeetCode](https://leetcode.com/problems/range-addition/) |
| **Find Pivot Index** | Compute left & right sums using prefix sum | [LeetCode](https://leetcode.com/problems/find-pivot-index/) |

🔗 **Reference for Prefix Sum & Difference Array:** [Click Here](https://leetcode.com/discuss/study-guide/1193713/Prefix-Sum-and-Difference-Array-CompleteStudy)

---

## **17. Trie (Prefix Tree)**  

### **What is a Trie?**  
A **Trie (Prefix Tree)** is a tree-like data structure used for efficient retrieval of words stored in a dataset. It is commonly used for **autocomplete, spell checking, and dictionary implementations.**  

Each node in a Trie represents a character, and words are stored as paths in the tree.  

### **Example Trie Structure**
```
Insert: ["cat", "cap", "bat"]
         (root)
        /      \
      c         b
     /  \        \
    a    a        a
   /      \        \
  t        p        t
```

### **Operations in Trie**
| Operation | Description | Time Complexity |
|-----------|------------|----------------|
| Insert a word | Add word character by character | **O(N)** |
| Search a word | Check if word exists in Trie | **O(N)** |
| Delete a word | Remove word from Trie | **O(N)** |
| Prefix search | Find words with a given prefix | **O(N)** |


### **Trie Algorithm (Pseudocode)**
```plaintext
1. Create a TrieNode class with:
   - children (array/hashmap of characters)
   - isEnd (boolean to mark end of word)
2. Insert a word:
   - Start from root, iterate over characters
   - Create new nodes if character not found
   - Mark last character as end of word
3. Search a word:
   - Traverse Trie following characters
   - If all characters found and isEnd is true, return True
4. Delete a word:
   - Traverse and unmark isEnd
   - Delete unnecessary nodes if no other words depend on them
```

#### **Java Implementation**
```java
class TrieNode {
    TrieNode[] children;
    boolean isEnd;

    public TrieNode() {
        children = new TrieNode[26]; // Assuming only lowercase a-z
        isEnd = false;
    }
}

class Trie {
    private TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    // Insert a word into the trie
    public void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) {
                node.children[index] = new TrieNode();
            }
            node = node.children[index];
        }
        node.isEnd = true;
    }

    // Search for a word in the trie
    public boolean search(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) return false;
            node = node.children[index];
        }
        return node.isEnd;
    }

    // Check if a prefix exists in the trie
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c : prefix.toCharArray()) {
            int index = c - 'a';
            if (node.children[index] == null) return false;
            node = node.children[index];
        }
        return true;
    }

    public static void main(String[] args) {
        Trie trie = new Trie();
        trie.insert("cat");
        trie.insert("cap");
        trie.insert("bat");

        System.out.println(trie.search("cat"));  // true
        System.out.println(trie.search("bat"));  // true
        System.out.println(trie.search("cap"));  // true
        System.out.println(trie.search("can"));  // false
        System.out.println(trie.startsWith("ca")); // true
    }
}
```

#### **Python Implementation**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    # Insert a word into the Trie
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    # Search for a word in the Trie
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    # Check if a prefix exists
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Example usage
trie = Trie()
trie.insert("cat")
trie.insert("cap")
trie.insert("bat")

print(trie.search("cat"))   # True
print(trie.search("bat"))   # True
print(trie.search("cap"))   # True
print(trie.search("can"))   # False
print(trie.starts_with("ca")) # True
```


### **Applications of Trie**
| Problem | Explanation | Link |
|---------|------------|------|
| **Autocomplete System** | Find words based on a prefix | [LeetCode](https://leetcode.com/problems/design-add-and-search-words-data-structure/) |
| **Spell Checker** | Check if a word exists | [LeetCode](https://leetcode.com/problems/implement-trie-prefix-tree/) |
| **Longest Word in Dictionary** | Find longest word that can be built letter by letter | [LeetCode](https://leetcode.com/problems/longest-word-in-dictionary/) |

🔗 **Reference for Trie (Prefix Tree):** [Click Here](https://leetcode.com/discuss/study-guide/1129847/Ultimate-Guide-to-Trie)

---

This completes all **17** pattern topics! 🎉 
