# LeetCode Patterns Guide

These patterns will help you avoid solving thousands of LeetCode problems. Instead, understanding these patterns will enable you to solve most problems with a structured approach.  
_Bhai, ek baar pattern samajh jao, baaki problems automatically handle ho jayengi!_ 🚀

---

## Table of Patterns

| No | Pattern Name                                | Example Problem / Link |
|----|---------------------------------------------|-------------------------|
| 1  | Sliding Window (Substring Problems)         | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) |
| 2  | Sliding Window (Frequency Pattern)          | [Frequency of the Most Frequent Element](https://leetcode.com/problems/frequency-of-the-most-frequent-element/) |
| 3  | Two Pointers                                | [Two Sum II - Sorted Array](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) |
| 4  | Backtracking (General)                      | [Backtracking Pattern](https://medium.com/leetcode-patterns/leetcode-pattern-3-backtracking-5d9e5a03dc26) |
| 5  | Backtracking (Permutations)                 | [Permutations](https://leetcode.com/problems/permutations/) |
| 6  | Dynamic Programming (DP)                    | [DP Patterns](https://leetcode.com/discuss/study-guide/458695/Dynamic-Programming-Patterns) |
| 7  | Dynamic Programming (DP II)                 | [DP Patterns II](https://leetcode.com/discuss/study-guide/1437879/Dynamic-Programming-Patterns) |
| 8  | Binary Search                               | [Binary Search Template](https://leetcode.com/discuss/study-guide/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems) |
| 9  | Graph Traversal (DFS/BFS)                     | [Graph Traversal](https://leetcode.com/discuss/study-guide/937307/lterative-or-Recursive-ar-DFS-and-BFS-Tree-Traversal-or-In-Pre-Post-and-LevelOrder-of-Viet) |
| 10 | Graph for Beginners                         | [Graph For Beginners](https://lectcode.com/discuss/study-guide/655708/Graph-For-Beginners-Problems-or-Pattern-or-Sample-Solutions) |
| 11 | Monotonic Stack                             | [Monotonic Stack](https://leetcode.com/discuss/study-guide/2347639/A-comprehensive-guide-and-template-for-monotonic-stack-based-problems) |
| 12 | Important String Patterns                   | [Important String Questions](https://leetcode.com/discuss/study-guide/2001789/Collections-of-Important-String-questions-Pattern) |
| 13 | Bits Manipulation                           | [Bits Manipulation](https://leetcode.com/discuss/interview-question/3695233/All-Types-of-Patterns-for-Bits-Manipulations-and-How-to-use-it) |
| 14 | BFS Pattern                                 | [BFS Pattern](https://medium.com/leetcode-patterns/leetcode-pattern-1-bfs-dis-25-of-the-problems-part-1-51945038-4353) |
| 15 | Greedy / Interval Pattern                   | [Greedy Interval Pattern](https://lectcode.com/discuss/general-discussion/794725/General-Pattern-for-greedy-approach-for-Interval-based-problems) <br> [Alternate Link](https://medium.com/setimpark0807/leetcode-is-easy-the-interval-pattern-d68a7c1c841) |

---

Below, each pattern is explained in detail with its theory, general template (step-by-step), and a sample problem (with question, intuition, and approach) along with Java and Python solutions (with explanatory comments). Related problems are also listed where applicable.

---

## 1. Sliding Window (Substring Problems)

### What is it?
Sliding Window is ek technique jo help karti hai problems solve karne me jahan hum continuous elements (subarray ya substring) dekh rahe hote hain. Isme hum ek window maintain karte hain jo gradually expand ya contract hoti hai instead of using nested loops.

### When to Apply?
- Jab problem me "longest", "smallest", "contiguous", "substring", "subarray" aata ho.
- Jab hume ek range ya segment me maximum ya minimum value dhoondni ho.

### General Template & DS/Algo:
1. **Data Structure**: Array, String, HashMap/Counter (to track frequency).
2. **Algo**: Maintain two pointers (`left` & `right`), expand window until condition satisfy ho, phir shrink karte hain to optimize result.
3. **Step-by-Step**:
   - Initialize left pointer to 0.
   - For each right pointer increment, update the window data (like frequency).
   - Check if current window meets condition (e.g., contains all required characters).
   - Once condition met, move left pointer to try to shrink the window.
   - Update result if a smaller valid window is found.

### Sample Problem: Minimum Window Substring

#### Question:
> **Given** two strings `s` and `t`, find the smallest substring in `s` that contains all characters of `t`. Agar valid substring nahi milti, return `""`.

#### How We Identified the Pattern:
- **Smallest substring** required jo all characters of `t` ko contain kare.
- Dynamic window size: expand jab tak condition satisfy na ho, phir contract to minimize size.

#### Approach & Intuition:
- **Expand Window**: Right pointer se characters add karo aur count track karo.
- **Contract Window**: Jab window valid ho (sab characters mil jaye), left pointer move karo to remove extra characters and update minimum length.
- **Use Data Structure**: Array (or HashMap) for frequency counting.

#### Java Code:
```java
class Solution {
    public String minWindow(String s, String t) {
        // Base case: agar s ya t empty ho toh return ""
        if (s.length() == 0 || t.length() == 0) return "";
        
        int[] map = new int[128]; // Frequency map for characters in t
        for (char c : t.toCharArray()) 
            map[c]++;
        
        int left = 0, right = 0, minLeft = 0, minLen = Integer.MAX_VALUE;
        int count = t.length(); // total characters needed
        
        while (right < s.length()) {
            // Expand window: include s[right] character
            if (map[s.charAt(right)] > 0) 
                count--; // one needed character found
            map[s.charAt(right)]--; // reduce frequency count
            right++;
            
            // If window contains all characters, try to contract from left
            while (count == 0) {
                // Update minimum length if current window is smaller
                if (right - left < minLen) {
                    minLeft = left;
                    minLen = right - left;
                }
                map[s.charAt(left)]++; // remove s[left] from window
                if (map[s.charAt(left)] > 0) 
                    count++; // one required character is missing now
                left++;
            }
        }
        // Return result if found, else empty string
        return minLen == Integer.MAX_VALUE ? "" : s.substring(minLeft, minLeft + minLen);
    }
}
```

#### Python Code:
```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # Base case check
        if not s or not t:
            return ""
        
        from collections import Counter
        countT = Counter(t)  # Frequency map for t
        left, right = 0, 0
        minLen = float('inf')
        minLeft = 0
        needed = len(t)  # total characters needed
        
        window = {}
        while right < len(s):
            char = s[right]
            window[char] = window.get(char, 0) + 1
            # If current char is needed and count is within limit, reduce needed
            if char in countT and window[char] <= countT[char]:
                needed -= 1
            
            # When window is valid, try to minimize it
            while needed == 0:
                if right - left + 1 < minLen:
                    minLeft, minLen = left, right - left + 1
                window[s[left]] -= 1
                # If removal of char breaks the condition, increase needed
                if s[left] in countT and window[s[left]] < countT[s[left]]:
                    needed += 1
                left += 1
            right += 1
        
        return "" if minLen == float('inf') else s[minLeft:minLeft + minLen]
```

#### Related Problems:
- [Frequency of the Most Frequent Element](https://leetcode.com/problems/frequency-of-the-most-frequent-element/)
- [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

---

## 2. Sliding Window (Frequency Pattern)

### What is it?
Yeh bhi sliding window ka hi ek variation hai. Yahan hum window ke andar frequency counts ya sums ko optimize karke solve karte hain.  
_Bhai, jab bhi frequency ya count calculate karna ho window ke andar, yeh template super useful hai!_

### When to Apply?
- Jab problem me continuous window ke andar frequency, sum, ya average calculate karna ho.
- Keywords: "frequency", "sum", "maximum", "minimum", "window".

### General Template & DS/Algo:
1. **DS**: HashMap/Counter to store frequency; Array for fixed-size cases.
2. **Steps**:
   - Expand window: update frequency count.
   - Check if window meets condition.
   - Contract window to optimize the answer.
3. **Extra Tip**: Sometimes sliding window is combined with greedy or binary search.

### Sample Problem: Frequency of the Most Frequent Element

_(Note: This problem asks to maximize the frequency of an element by performing at most k operations.)_

#### Question:
> **Given** an array and an integer k, find the maximum frequency of any element after performing at most k increments on array elements.

#### How We Identified the Pattern:
- Window size grows until the cost (number of operations) exceeds k.
- When it exceeds k, shrink the window from the left.
- Typical sliding window with additional condition on sum/difference.

#### Approach & Intuition:
- **Expand**: Right pointer adds element and update total operations needed.
- **Contract**: If cost > k, move left pointer to reduce cost.
- **Result**: Maximum window size gives maximum frequency.

#### Java Code (Template-style):
```java
class Solution {
    public int maxFrequency(int[] nums, int k) {
        Arrays.sort(nums); // sorted array for easier calculation
        int left = 0, total = 0, res = 1;
        for (int right = 0; right < nums.length; right++) {
            total += nums[right];
            // Calculate cost: needed operations = nums[right]*(window size) - total sum
            while (nums[right] * (right - left + 1) - total > k) {
                total -= nums[left];
                left++;
            }
            res = Math.max(res, right - left + 1);
        }
        return res;
    }
}
```

#### Python Code:
```python
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        total = 0
        res = 1
        for right in range(len(nums)):
            total += nums[right]
            # While operations required exceed k, shrink the window
            while nums[right] * (right - left + 1) - total > k:
                total -= nums[left]
                left += 1
            res = max(res, right - left + 1)
        return res
```

#### Related Problems:
- [Longest Subarray with Sum Constraint](#)
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

---

## 3. Two Pointers Pattern

### What is it?
Two pointers technique use karte hain jab array sorted ho ya jab hume pair/triplet check karna ho.  
_Socho, ek pointer start se aur ek end se, dono ko adjust karke target achieve karna._

### When to Apply?
- Jab problem me sorted array hai.
- Keywords: "pair", "triplet", "sorted", "remove duplicates", "target sum".

### General Template & DS/Algo:
1. **DS**: Array (sorted)
2. **Algo**: Set `left = 0` and `right = n-1`, then adjust based on condition.
3. **Step-by-Step**:
   - Calculate sum of elements at left and right.
   - Agar sum target se kam ho, increment left.
   - Agar sum target se zyada ho, decrement right.
   - Return indices when condition match.

### Sample Problem: Two Sum II - Sorted Array

#### Question:
> **Given** a 1-indexed sorted array and a target, return indices of two numbers such that they add up to the target.

#### How We Identified the Pattern:
- Sorted array se direct two pointers lagate hain.
- Avoid karte hain nested loops by adjusting pointers.

#### Approach & Intuition:
- **Start**: left pointer at beginning, right pointer at end.
- **Check**: Sum = numbers[left] + numbers[right].  
   - Agar equal ho, return result.
   - Agar sum chota, increment left.
   - Agar bada, decrement right.

#### Java Code:
```java
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            // Check if we found the target sum
            if (sum == target) 
                return new int[]{left + 1, right + 1}; // 1-indexed
            else if (sum < target) 
                left++; // Increase sum by moving left pointer rightwards
            else 
                right--; // Decrease sum by moving right pointer leftwards
        }
        return new int[]{}; // In case no solution is found
    }
}
```

#### Python Code:
```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            curr_sum = numbers[left] + numbers[right]
            if curr_sum == target:
                return [left + 1, right + 1]  # 1-indexed result
            elif curr_sum < target:
                left += 1  # Move left pointer to increase sum
            else:
                right -= 1  # Move right pointer to decrease sum
        return []  # If no solution is found
```

#### Related Problems:
- [3Sum](https://leetcode.com/problems/3sum/)
- [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

---

## 4. Backtracking (General)

### What is it?
Backtracking ek recursive approach hai jisme hum sabhi possible choices explore karte hain aur agar koi choice sahi na ho toh backtrack karke next option try karte hain.  
_Yeh approach tab use karo jab hume permutations, combinations ya subsets generate karne ho._

### When to Apply?
- Jab problem me “all possible combinations” ya “all valid arrangements” dhoondhne ho.
- Keywords: "generate all", "combinations", "permutations", "backtrack".

### General Template & DS/Algo:
1. **DS**: List to store temporary results.
2. **Algo**:
   - Recursively add choices.
   - Agar condition violate ho, undo (backtrack) the last step.
3. **Step-by-Step**:
   - Start with an empty temporary list.
   - For each element starting from a given index, add it to the list.
   - Recursively call the function.
   - Remove the last element (backtrack) and try next.

### Sample Problem: (Reference from Medium article) Backtracking Pattern

#### Question:
> **Given** a set of numbers, generate all possible subsets (or combinations).  
_(Example: For `[1, 2, 3]`, return all subsets.)_

#### How We Identified the Pattern:
- Problem me sabhi possible subsets generate karne hai → backtracking best fit hai.
- Har step me choice lene ka decision, aur phir backtrack karna.

#### Approach & Intuition:
- **Recursive function**: Har recursion me current element choose karo, phir recursion call karke baaki elements consider karo.
- **Backtrack**: Agar combination complete ho ya invalid ho, remove last element and try next.

#### Java Code:
```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res, new ArrayList<>(), nums, 0);
        return res;
    }
    private void backtrack(List<List<Integer>> res, List<Integer> tempList, int[] nums, int start) {
        // Add current combination to result
        res.add(new ArrayList<>(tempList));
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);          // Choose element nums[i]
            backtrack(res, tempList, nums, i + 1); // Recurse with updated list
            tempList.remove(tempList.size() - 1);  // Backtrack: remove last element
        }
    }
}
```

#### Python Code:
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(start, path):
            res.append(path[:])  # Append a copy of current combination
            for i in range(start, len(nums)):
                path.append(nums[i])      # Choose element nums[i]
                backtrack(i + 1, path)      # Recurse for next elements
                path.pop()                  # Backtrack: remove last element
        backtrack(0, [])
        return res
```

#### Related Problems:
- [Combination Sum](https://leetcode.com/problems/combination-sum/)
- [Word Search](https://leetcode.com/problems/word-search/)

---

## 5. Backtracking (Permutations)

### What is it?
Yeh bhi backtracking ka hi ek form hai, jisme hum specifically **permutations** generate karte hain.  
_Bhai, jab order matter karta ho, tab permutations ka use karo._

### When to Apply?
- Jab problem me saare possible orderings ya arrangements chahiye ho.
- Keywords: "permutations", "order", "arrangements".

### General Template & DS/Algo:
1. **DS**: List for temporary permutation, boolean array for tracking usage.
2. **Algo**:
   - Recursive function jo har position ke liye unused elements choose kare.
   - Backtrack after exploring each possibility.
3. **Step-by-Step**:
   - Initialize an empty list and a visited array.
   - For each unvisited element, mark it as visited and add to list.
   - Recursively call for remaining positions.
   - Backtrack by unmarking element and removing it.

### Sample Problem: Permutations

#### Question:
> **Given** an array of distinct integers, return all possible permutations.

#### How We Identified the Pattern:
- Order of elements matter karta hai → use permutations backtracking.
- Har recursive call me ek element choose karo, aur backtrack karte hue all orders generate karo.

#### Approach & Intuition:
- **Recursive Depth-first search (DFS)**: Har call me ek element add karo.
- **Backtracking**: Agar current permutation complete ho gaya, add to result; else backtrack to try next element.

#### Java Code:
```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res, new ArrayList<>(), nums);
        return res;
    }
    private void backtrack(List<List<Integer>> res, List<Integer> tempList, int[] nums) {
        if (tempList.size() == nums.length) {
            res.add(new ArrayList<>(tempList));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (tempList.contains(nums[i])) continue; // Skip if already used
            tempList.add(nums[i]);               // Choose element
            backtrack(res, tempList, nums);        // Recurse for next element\n            tempList.remove(tempList.size() - 1); // Backtrack\n        }\n    }\n}\n```

#### Python Code:
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(path):
            if len(path) == len(nums):
                res.append(path[:])
                return
            for num in nums:
                if num in path:
                    continue  # Skip already used numbers
                path.append(num)      # Choose element
                backtrack(path)       # Recurse
                path.pop()            # Backtrack
        backtrack([])
        return res
```

#### Related Problems:
- [Subsets](https://leetcode.com/problems/subsets/)
- [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

---

## 6. Dynamic Programming (DP)

### What is it?
Dynamic Programming ek technique hai jisme hum complex problems ko chote subproblems me todte hain aur unke solutions ko store karke reuse karte hain.  
_Bhai, agar overlapping subproblems ho aur optimal substructure ho, toh DP try karo!_

### When to Apply?
- Keywords: "minimum cost", "maximum sum", "number of ways", "longest sequence".
- Problems where recursive solutions overlap and can be memoized.

### General Template & DS/Algo:
1. **DS**: Array or HashMap for memoization.
2. **Algo**:
   - Define state and recurrence relation.
   - Use recursion with memoization or tabulation.
3. **Step-by-Step**:
   - Identify base case(s).
   - Write recurrence: dp[i] = function(dp[...]).
   - Loop or recursion to fill dp table.

### Sample Problem: (Example – Climbing Stairs)

#### Question:
> **Given** n stairs, each time you can climb 1 or 2 steps. Return the number of distinct ways to climb to the top.

#### How We Identified the Pattern:
- Overlapping subproblems (f(n) = f(n-1) + f(n-2)) → classic DP.
- Optimal substructure exists.

#### Approach & Intuition:
- **Recurrence**: f(n) = f(n-1) + f(n-2)
- **Tabulation**: Build up the solution from base cases.

#### Java Code:
```java
class Solution {
    public int climbStairs(int n) {
        if(n <= 2) return n;
        int[] dp = new int[n+1];
        dp[1] = 1; dp[2] = 2;
        for(int i = 3; i <= n; i++){
            dp[i] = dp[i-1] + dp[i-2]; // Current ways = ways to previous two stairs
        }
        return dp[n];
    }
}
```

#### Python Code:
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        dp = [0] * (n + 1)
        dp[1], dp[2] = 1, 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]  # Recurrence relation
        return dp[n]
```

#### Related Problems:
- [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- [Coin Change](https://leetcode.com/problems/coin-change/)

---

## 7. Dynamic Programming (DP II)

### What is it?
Yeh DP ka advanced form hai jahan state space complex hota hai.  
_Bhai, jab 2D DP ya more complex recurrence relations ho, tab use karte hain._

### When to Apply?
- Complex state definitions.
- Keywords: "matrix", "grid", "optimal path", "advanced DP".

### General Template & DS/Algo:
- **DS**: 2D arrays, multi-dimensional DP tables.
- **Steps**: Define dp[i][j] based on previous computed states.
- **Example**: Longest Common Subsequence, Edit Distance.

_(For brevity, sample code is similar to standard 2D DP problems. Refer to discussion link for detailed templates.)_

#### Related Problems:
- [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [Edit Distance](https://leetcode.com/problems/edit-distance/)

---

## 8. Binary Search Pattern

### What is it?
Binary Search use karte hain jab array sorted ho ya answer kisi range me ho aur hume efficiently search karna ho.  
_Bhai, O(log n) time me answer mil jaata hai, so use binary search when possible!_

### When to Apply?
- Keywords: "sorted", "search", "target", "range", "log n".
- Problems where decision function can be applied.

### General Template & DS/Algo:
1. **DS**: Sorted array.
2. **Algo**:
   - Set left and right pointers.
   - While left <= right, compute mid, decide which half to search.
3. **Step-by-Step**:
   - Initialize left = 0, right = n-1.
   - While loop with condition, update boundaries based on comparison.

#### Sample Problem: Binary Search Template

_(Refer to provided link for a robust Python template.)_

#### Java Code:
```java
class Solution {
    public int binarySearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) return mid;
            else if(nums[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1; // Not found
    }
}
```

#### Python Code:
```python
class Solution:
    def binarySearch(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1  # Target not found
```

#### Related Problems:
- [Find First and Last Position of Element](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Search Insert Position](https://leetcode.com/problems/search-insert-position/)

---

## 9. Graph Traversal (DFS/BFS)

### What is it?
Graph traversal techniques like DFS (Depth-First Search) and BFS (Breadth-First Search) are used for exploring graphs, trees, or grids.  
_Bhai, jab nodes ya cells se connected problems ho, DFS/BFS dono ka use karo._

### When to Apply?
- Keywords: "graph", "tree", "grid", "traverse", "shortest path".
- Problems like number of islands, maze problems, etc.

### General Template & DS/Algo:
1. **DS**: Queue for BFS, Stack/recursion for DFS.
2. **Algo**:
   - For BFS: use a queue, mark visited nodes.
   - For DFS: use recursion or stack to explore deeply.
3. **Step-by-Step** (BFS Example):
   - Initialize queue with start node.
   - While queue not empty, process node and enqueue unvisited neighbors.

#### Sample Problem: Graph Traversal Template (BFS)

_(For detailed code, refer to provided link.)_

#### Java Code (BFS):
```java
class Solution {
    public void bfs(Node start) {
        Queue<Node> queue = new LinkedList<>();
        Set<Node> visited = new HashSet<>();
        queue.add(start);
        visited.add(start);
        while(!queue.isEmpty()){
            Node node = queue.poll();
            // Process node
            for(Node neighbor : node.neighbors){
                if(!visited.contains(neighbor)){
                    queue.add(neighbor);
                    visited.add(neighbor);
                }
            }
        }
    }
}
```

#### Python Code (BFS):
```python
class Solution:
    def bfs(self, start: Node) -> None:
        from collections import deque
        queue = deque([start])
        visited = {start}
        while queue:
            node = queue.popleft()
            # Process node
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
```

#### Related Problems:
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [Word Ladder](https://leetcode.com/problems/word-ladder/)

---

## 10. Graph for Beginners

### What is it?
Yeh basic graph problems ke liye ek simplified template hai, jisme beginners ko DFS/BFS samajhne me madad milti hai.  
_Easy-to-understand approach for tree or graph traversal with clear examples._

### When to Apply?
- Jab graph problems mein new ho ya simple traversal chahiye.
- Keywords: "graph basics", "beginner", "traverse".

### General Template:
- Similar to above DFS/BFS, with extra comments and simpler structure.

_(Due to similarity, refer to section 9 for detailed code, and modify as per problem requirements.)_

#### Related Problems:
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

---

## 11. Monotonic Stack Pattern

### What is it?
Monotonic Stack ek technique hai jisme hum stack maintain karte hain in increasing or decreasing order.  
_Yeh tab use hota hai jab hume next greater or smaller element dhoondna ho efficiently._

### When to Apply?
- Keywords: "next greater element", "stock span", "histogram".
- Problems requiring immediate larger/smaller element lookup.

### General Template & DS/Algo:
1. **DS**: Stack.
2. **Algo**:
   - Iterate through array.
   - While stack is not empty and current element is greater than element at stack top, pop stack.
   - Push current index.
3. **Step-by-Step**:
   - For each element, check and pop all elements smaller than current.
   - Process popped elements as required.

#### Sample Problem: Monotonic Stack Template

_(Refer to provided link for detailed variations.)_

#### Java Code:
```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < 2 * n; i++) {
            while (!stack.isEmpty() && nums[i % n] > nums[stack.peek()]) {
                res[stack.pop()] = nums[i % n];
            }
            if (i < n) stack.push(i);
        }
        return res;
    }
}
```

#### Python Code:
```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [-1] * n
        stack = []
        for i in range(2 * n):
            while stack and nums[i % n] > nums[stack[-1]]:
                res[stack.pop()] = nums[i % n]
            if i < n:
                stack.append(i)
        return res
```

#### Related Problems:
- [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

---

## 12. Important String Patterns

### What is it?
Yeh collection of string-based patterns hai jo bahut saare important string questions me repeat hote hain.  
_Agar string manipulation, palindrome check, ya substring search ho, toh yeh guide kaam aata hai._

### When to Apply?
- Keywords: "string", "palindrome", "substring", "pattern matching".
- Use when problems require detailed string processing.

### General Template & DS/Algo:
- **DS**: Two pointers, HashMap, or even stacks depending on sub-problem.
- **Approach**: Combine techniques like sliding window or two pointers with string-specific checks.

#### Sample Problem:
_(Example: Longest Palindromic Substring – use expand around center technique.)_

_(Due to brevity, refer to the guide in the link for detailed approaches.)_

#### Related Problems:
- [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
- [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

---

## 13. Bits Manipulation Pattern

### What is it?
Bits Manipulation techniques involve using bitwise operations to solve problems fast and efficiently.  
_Yeh tab use hota hai jab numbers ko binary representation me process karna ho._

### When to Apply?
- Keywords: "bitwise", "XOR", "set bits", "binary".
- Problems where arithmetic operations can be optimized using bits.

### General Template & DS/Algo:
1. **DS**: No extra DS required; just use bitwise operators.
2. **Approach**:
   - Use operators like &, |, ^, <<, >> to manipulate bits.
   - Common operations: checking power of two, counting bits, etc.
3. **Step-by-Step**:
   - Identify binary pattern in problem.
   - Apply bit masks and shifts accordingly.

#### Sample Problem: Single Number

#### Question:
> **Given** a non-empty array of integers where every element appears twice except for one, find that single one.

#### How We Identified the Pattern:
- XOR operator properties: a ^ a = 0, so repeatedly XOR all numbers gives the single number.

#### Java Code:
```java
class Solution {
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;  // XOR accumulates to single number
        }
        return result;
    }
}
```

#### Python Code:
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result ^= num  # XOR property used here
        return result
```

#### Related Problems:
- [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)
- [Missing Number](https://leetcode.com/problems/missing-number/)

---

## 14. BFS Pattern

### What is it?
Breadth-First Search (BFS) ek graph/tree traversal technique hai jisme hum level by level explore karte hain.  
_Yeh pattern useful hai jab shortest path ya minimum moves dhoondhne ho._

### When to Apply?
- Keywords: "BFS", "shortest path", "level order", "graph", "tree".
- Use when you need to traverse or search layer-wise.

### General Template & DS/Algo:
1. **DS**: Queue.
2. **Approach**:
   - Enqueue the start node.
   - Process nodes level by level.
3. **Step-by-Step**:
   - While queue not empty, dequeue node.
   - Process its neighbors and mark them as visited.
   - Enqueue unvisited neighbors.

#### Sample Problem: (BFS Template from the link)

#### Java Code:
```java
class Solution {
    public void bfs(Node start) {
        Queue<Node> queue = new LinkedList<>();
        Set<Node> visited = new HashSet<>();
        queue.add(start);
        visited.add(start);
        while (!queue.isEmpty()) {
            Node node = queue.poll();
            // Process the current node (e.g., print or check condition)
            for (Node neighbor : node.neighbors) {
                if (!visited.contains(neighbor)) {
                    queue.add(neighbor);
                    visited.add(neighbor);
                }
            }
        }
    }
}
```

#### Python Code:
```python
class Solution:
    def bfs(self, start: Node) -> None:
        from collections import deque
        queue = deque([start])
        visited = {start}
        while queue:
            node = queue.popleft()
            # Process current node
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
```

#### Related Problems:
- [Word Ladder](https://leetcode.com/problems/word-ladder/)
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

---

## 15. Greedy / Interval Pattern

### What is it?
Greedy algorithm ka use tab hota hai jab local optimal choice lene se overall optimum solution mil jata hai. Interval problems me yeh approach kaafi common hai.  
_Bhai, jab intervals ya scheduling problems ho, greedy approach se kaam ban jata hai!_

### When to Apply?
- Keywords: "interval", "greedy", "schedule", "non-overlapping".
- Use when making a local optimal choice gives the best global result.

### General Template & DS/Algo:
1. **DS**: Sort intervals by start or end times.
2. **Approach**:
   - Sort intervals.
   - Iterate through intervals and choose the one with the earliest finish time (for maximum non-overlap) or according to problem condition.
3. **Step-by-Step**:
   - Sort the intervals.
   - Initialize a variable to keep track of the last chosen interval's end.
   - Iterate and select intervals that do not conflict.

#### Sample Problem: Greedy Interval Pattern

_(For example, consider the Non-overlapping Intervals problem.)_

#### Java Code:
```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) return 0;
        // Sort intervals based on end time
        Arrays.sort(intervals, (a, b) -> a[1] - b[1]);
        int count = 0;
        int end = intervals[0][1];
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] < end) {
                count++;  // Overlap found, increment removal count
            } else {
                end = intervals[i][1];  // Update end to current interval's end
            }
        }
        return count;
    }
}
```

#### Python Code:
```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[1])
        count = 0
        end = intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] < end:
                count += 1
            else:
                end = intervals[i][1]
        return count
```

#### Related Problems:
- [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

---

_That's a wrap for all 15 patterns!_  
