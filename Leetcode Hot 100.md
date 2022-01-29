## 算法思维 

- 基础技巧：分治、二分、贪心
- 排序算法：快速排序、归并排序、计数排序
- 搜索算法：回溯、递归、深度优先遍历，广度优先遍历，二叉搜索树等
- 图论：最短路径、最小生成树
- 动态规划：背包问题、最长子序列

### 动态规划
##### [312\. 戳气球](https://leetcode-cn.com/problems/burst-balloons/)
```java
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> rec(n + 2, vector<int>(n + 2));
        vector<int> val(n + 2);
        val[0] = val[n + 1] = 1;
        for (int i = 1; i <= n; i++) {
            val[i] = nums[i - 1];
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 2; j <= n + 1; j++) {
                for (int k = i + 1; k < j; k++) {
                    int sum = val[i] * val[k] * val[j];
                    sum += rec[i][k] + rec[k][j];
                    rec[i][j] = max(rec[i][j], sum);
                }
            }
        }
        return rec[0][n + 1];
    }
};

```

#### 股票问题

#### 背包问题

##### [322\. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)
```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int dp[][]=new int[coins.length][amount+1];

        for(int j=0;j<=amount;j++){
            if(j%coins[0]==0){
                dp[0][j]=j/coins[0];
            }else{
                dp[0][j]=amount+1;
            }
        }

        for(int i=0;i<coins.length;i++){
            dp[i][0]=0;
        }

        for(int i=1;i<coins.length;i++){
            for(int j=1;j<=amount;j++){
                if(coins[i]<=j){
                    dp[i][j]=Math.min(dp[i][j-coins[i]]+1,dp[i-1][j]);
                }else{
                    dp[i][j]=dp[i-1][j];
                }
            }
        }
        return (dp[coins.length-1][amount]==(amount+1))?-1:dp[coins.length-1][amount];
    }
}
```
##### [416\. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)
```java
class Solution {
    public boolean canPartition(int[] nums) {
        int target=Arrays.stream(nums).sum();
        if(target%2!=0) return false;

        target=target/2;

        int max=Arrays.stream(nums).max().getAsInt();
        if(max>target) return false;
        boolean[][] dp=new boolean[nums.length][target+1];
        dp[0][nums[0]]=true;
        for(int i=0;i< nums.length;i++){
            dp[i][0]=true;
        }
        for(int i=1;i<nums.length;i++){
            for(int j=1;j<=target;j++){
                if(nums[i]>j){
                    dp[i][j]=dp[i-1][j];
                }else{
                    dp[i][j]=dp[i-1][j]|dp[i-1][j-nums[i]];
                }
            }
        }

        return dp[nums.length-1][target];
    }
}
```
##### [494\. 目标和](https://leetcode-cn.com/problems/target-sum/)

```java
class Solution {
    int count = 0;

    public int findTargetSumWays(int[] nums, int target) {
        backtrack(nums, target, 0, 0);
        return count;
    }

    public void backtrack(int[] nums, int target, int index, int sum) {
        if (index == nums.length) {
            if (sum == target) {
                count++;
            }
        } else {
            backtrack(nums, target, index + 1, sum + nums[index]);
            backtrack(nums, target, index + 1, sum - nums[index]);
        }
    }
}
```

### 回溯

##### [494\. 目标和](https://leetcode-cn.com/problems/target-sum/)

```java
class Solution {
    int count = 0;

    public int findTargetSumWays(int[] nums, int target) {
        backtrack(nums, target, 0, 0);
        return count;
    }

    public void backtrack(int[] nums, int target, int index, int sum) {
        if (index == nums.length) {
            if (sum == target) {
                count++;
            }
        } else {
            backtrack(nums, target, index + 1, sum + nums[index]);
            backtrack(nums, target, index + 1, sum - nums[index]);
        }
    }
}
```
### 贪心

##### [45. 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

```java
class Solution {
    public int jump(int[] nums) {
        int length = nums.length;
        int end = 0;
        int maxPosition = 0; 
        int steps = 0;
        for (int i = 0; i < length - 1; i++) {
            maxPosition = Math.max(maxPosition, i + nums[i]); 
            if (i == end) {
                end = maxPosition;
                steps++;
            }
        }
        return steps;
    }
}
```

### 排序
##### [347\. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)
堆排序
```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        // int[] 的第一个元素代表数组的值，第二个元素代表了该值出现的次数
        PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] m, int[] n) {
                return m[1] - n[1];
            }
        });
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            if (queue.size() == k) {
                if (queue.peek()[1] < count) {
                    queue.poll();
                    queue.offer(new int[]{num, count});
                }
            } else {
                queue.offer(new int[]{num, count});
            }
        }
        int[] ret = new int[k];
        for (int i = 0; i < k; ++i) {
            ret[i] = queue.poll()[0];
        }
        return ret;
    }
}

```
快排
```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        List<int[]> values = new ArrayList<int[]>();
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            values.add(new int[]{num, count});
        }
        int[] ret = new int[k];
        qsort(values, 0, values.size() - 1, ret, 0, k);
        return ret;
    }

    public void qsort(List<int[]> values, int start, int end, int[] ret, int retIndex, int k) {
        int picked = (int) (Math.random() * (end - start + 1)) + start;
        Collections.swap(values, picked, start);
        
        int pivot = values.get(start)[1];
        int index = start;
        for (int i = start + 1; i <= end; i++) {
            if (values.get(i)[1] >= pivot) {
                Collections.swap(values, index + 1, i);
                index++;
            }
        }
        Collections.swap(values, start, index);

        if (k <= index - start) {
            qsort(values, start, index - 1, ret, retIndex, k);
        } else {
            for (int i = start; i <= index; i++) {
                ret[retIndex++] = values.get(i)[0];
            }
            if (k > index - start + 1) {
                qsort(values, index + 1, end, ret, retIndex, k - (index - start + 1));
            }
        }
    }
}

```


## 数据结构

- 数组与链表：单 / 双向链表
- 栈与队列
- 哈希表
- 堆：最大堆 ／ 最小堆
- 树与图：最近公共祖先、并查集
- 字符串：前缀树（字典树） ／ 后缀树

#### 线性表
```java
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if(root1==null)
            return root2;
        if(root2==null)
            return root1;
        TreeNode root=new TreeNode(root1.val+root2.val);
        root.left=mergeTrees(root1.left,root2.left);
        root.right=mergeTrees(root1.right,root2.right);
        return root;
        
    }
}
```
#### 二叉树
#### [337\. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
    class Solution {
        Map<TreeNode, Integer> f = new HashMap<TreeNode, Integer>();
        Map<TreeNode, Integer> g = new HashMap<TreeNode, Integer>();

        public int rob(TreeNode root) {
            dfs(root);
            return Math.max(f.getOrDefault(root, 0), g.getOrDefault(root, 0));
        }

        public void dfs(TreeNode node) {
            if (node == null) {
                return;
            }
            dfs(node.left);
            dfs(node.right);
            //f 选择node节点
            f.put(node, node.val + g.getOrDefault(node.left, 0) + g.getOrDefault(node.right, 0));
            //g 不选择 node节点，node 的子节点可选可不选
            g.put(node, Math.max(f.getOrDefault(node.left, 0), g.getOrDefault(node.left, 0)) + Math.max(f.getOrDefault(node.right, 0), g.getOrDefault(node.right, 0)));
        }
    }

```
##### [437\. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)
```java

import java.util.HashMap;
import java.util.Map;

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    Map<Integer,Integer> preSum=new HashMap<>();
    public int pathSum(TreeNode root, int sum) {
        preSum.put(0, 1);
        int result=bfs(root,sum,0);
        return result;
    }

    private int bfs(TreeNode root, int target, int curSum){

        if(root==null) return 0;
        int res=0;
        curSum=curSum+root.val;

        res += preSum.getOrDefault(curSum - target, 0);

        preSum.put(curSum, preSum.getOrDefault(curSum, 0) + 1);


        res += bfs(root.left, target, curSum);
        res += bfs(root.right,  target, curSum);

        preSum.put(curSum, preSum.get(curSum) - 1);
        return res;

    }
}

```
##### [617\. 合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if(root1==null)
            return root2;
        if(root2==null)
            return root1;
        TreeNode root=new TreeNode(root1.val+root2.val);
        root.left=mergeTrees(root1.left,root2.left);
        root.right=mergeTrees(root1.right,root2.right);
        return root;
    }
}
```

##### [538\. 把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int sum=0;
    public TreeNode convertBST(TreeNode root) {
        if(root!=null){
            convertBST(root.right);
            sum=sum+root.val;
            root.val=sum;
            convertBST(root.left);
        }
        return root;
    }
}

```
#### [543\. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int ans=0;
    public int diameterOfBinaryTree(TreeNode root) {

        dep(root);
        return ans-1;
    }
    private int dep(TreeNode root){
        if(root==null) return 0;
        int l=dep(root.left);
        int r=dep(root.right);
        int dia=l+r+1;
        ans=Math.max(dia,ans);
        return Math.max(l,r)+1;
    }
}

```


#### 栈

##### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```java
        public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
            if(l1==null) return l2;
            if(l2==null) return l1;
            if(l1.val<l2.val){
                l1.next=mergeTwoLists(l1.next,l2);
                return l1;
            }else{
                l2.next=mergeTwoLists(l1, l2.next);
                return l2;
            }
        }
```



#### 普通栈 

##### [速记内容还原](http://3ms.huawei.com/km/groups/3803117/blogs/details/8998831?l=zh-cn)

   ```java
       private String UnzipString(String records){
           String result="";
   
           LinkedList<StringBuilder> stack_res=new LinkedList<>();
           int cur_multi=0;
           StringBuilder cur=new StringBuilder();
           stack_res.push(cur);
           for(char record:records.toCharArray()){
               if(Character.isAlphabetic(record)) {
                   stack_res.peek().append(record);
               }
               if(record=='('){
                   cur=new StringBuilder();
                   stack_res.push(cur);
               }
               if(record==')'){
                   continue;
               }
               if(Character.isDigit(record)){
                   cur_multi=cur_multi*10+record-'0';
               }
               if(record=='<'){
                   cur_multi=0;
               }
               if(record=='>'){
                   StringBuilder temp=stack_res.pop();
                   for(int i=0;i<cur_multi;i++){
                       stack_res.peek().append(temp);
                   }
               }
           }
           return stack_res.peek().toString();
       }
   
   ```
##### [394\. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

```java
class Solution {
    public String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        LinkedList<Integer> stack_multi = new LinkedList<>();
        LinkedList<String> stack_res = new LinkedList<>();
        for(Character c : s.toCharArray()) {
            if(c == '[') {
                stack_multi.addLast(multi);
                stack_res.addLast(res.toString());
                multi = 0;
                res = new StringBuilder();
            }
            else if(c == ']') {
                StringBuilder tmp = new StringBuilder();
                int cur_multi = stack_multi.removeLast();
                for(int i = 0; i < cur_multi; i++) tmp.append(res);
                res = new StringBuilder(stack_res.removeLast() + tmp);
            }
            else if(c >= '0' && c <= '9') multi = multi * 10 + Integer.parseInt(c + "");
            else res.append(c);
        }
        return res.toString();
    }

}

```
##### [1190. 反转每对括号间的子串](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

```j'a
class Solution {
    public String reverseParentheses(String s) {

        LinkedList<StringBuilder> res_stack=new LinkedList<>();
        StringBuilder sb=new StringBuilder();
        for(int i=0;i<s.length();i++){
            if(Character.isAlphabetic(s.charAt(i))){
                sb.append(s.charAt(i));
                continue;
            }
            if(s.charAt(i)=='('){
                res_stack.push(sb);
                sb=new StringBuilder();
                continue;
            }
            if(s.charAt(i)==')'){
                StringBuilder temp=res_stack.pop();
                sb=temp.append(sb.reverse());
            }
        }
        return sb.toString();
    }
}
```



#### 单调栈

##### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)
单调栈 栈顶元素最小
```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        LinkedList<Integer> stack=new LinkedList<>();
        int[] result=new int[temperatures.length];
        for (int i=0;i< temperatures.length;i++){
            while((!stack.isEmpty())&&(temperatures[i]>temperatures[stack.peek()])){
                int index=stack.pop();
                result[index]=i-index;
            }
            stack.push(i);
        }
        return result;
    }
}
```



#### 哈希表

   ##### [任务规划](http://3ms.huawei.com/km/groups/3803117/blogs/details/9544197?l=zh-cn)

   ```java
       public int divideGroup(int[] tasks, int[][] mutexPairs) {
   
           int result=1;
   
           Map<Integer,Set<Integer>> map=new HashMap<>();
           Arrays.stream(tasks).forEach(task->map.put(task,new HashSet<>()));
           Arrays.stream(mutexPairs).forEach(ints -> map.get(ints[0]).add(ints[1]));
   
           Set<Integer> curSet=new HashSet<>();
   
           for(int i=0;i< tasks.length;i++){
               if(curSet.contains(tasks[i])){
                   result++;
                   curSet.clear();
                   curSet.addAll(map.get(tasks[i]));
               }else{
                   curSet.addAll(map.get(tasks[i]));
               }
           }
           return result;
       }
   ```

#### 链表

##### [143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

```java
class Solution {
    public void reorderList(ListNode head) {
        ListNode middle=findMiddle(head);
        ListNode head2=reverseList(middle.next);
        middle.next=null;
        mergeList(head,head2);
    }

    private ListNode reverseList(ListNode listNode){
        ListNode pre=null;
        ListNode cur=listNode;
        while(cur!=null){
            ListNode next=cur.next;
            cur.next=pre;
            pre=cur;
            cur=next;
        }
        return pre;
    }


    private ListNode findMiddle(ListNode listNode){
        ListNode slow=listNode;
        ListNode fast=listNode;
        while(fast!=null&&fast.next!=null){
            slow=slow.next;
            fast=fast.next.next;
        }
        return slow;
    }

    public void mergeList(ListNode l1, ListNode l2) {
        ListNode l1_tmp;
        ListNode l2_tmp;
        while (l1 != null && l2 != null) {
            l1_tmp = l1.next;
            l2_tmp = l2.next;

            l1.next = l2;
            l1 = l1_tmp;

            l2.next = l1;
            l2 = l2_tmp;
        }
    }
}
```

## 技巧

#### 双指针

##### [581\. 最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        int maxn = Integer.MIN_VALUE, right = -1;
        int minn = Integer.MAX_VALUE, left = -1;
        for (int i = 0; i < n; i++) {
            if (maxn > nums[i]) { //maxn:表示前一项;nums[i]:表示当前项
                right = i;//可理解为:前一项比当前项大时,该数组不为升序数组,并记录当前项.  遍历一次后,right即为最后一个使之不为升序数组的数.  left同理
            } else {
                maxn = nums[i];
            }
            if (minn < nums[n - i - 1]) {
                left = n - i - 1;
            } else {
                minn = nums[n - i - 1];
            }
        }
        return right == -1 ? 0 : right - left + 1;
    }
}
```
#### 前缀树

##### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

#### 前缀和

##### [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/)

##### [560\. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)
```java
import java.util.HashMap;
import java.util.Map;

public class Solution {

    public int subarraySum(int[] nums, int k) {
        // key：前缀和，value：key 对应的前缀和的个数
        Map<Integer, Integer> preSumFreq = new HashMap<>();
        // 对于下标为 0 的元素，前缀和为 0，个数为 1
        preSumFreq.put(0, 1);

        int preSum = 0;
        int count = 0;
        for (int num : nums) {
            preSum += num;

            // 先获得前缀和为 preSum - k 的个数，加到计数变量里
            if (preSumFreq.containsKey(preSum - k)) {
                count += preSumFreq.get(preSum - k);
            }

            // 然后维护 preSumFreq 的定义
            preSumFreq.put(preSum, preSumFreq.getOrDefault(preSum, 0) + 1);
        }
        return count;
    }
}
```

#### 滑动窗口

##### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int maxLen=0,r=0;
        Set<Character> set=new HashSet<>();
        for(int l=0;l<s.length();l++){
            set.add(s.charAt(l));
            while(set.size()<l-r+1){
                if(s.charAt(r)!=s.charAt(l)){
                    set.remove(s.charAt(r));
                }
                r++;
            }
            maxLen=Math.max(maxLen,l-r+1);
        }
        return maxLen;
    }
}
```

##### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int min=nums.length;
        for(int r=0,l=0;r< nums.length;r++){
            while(sum(nums,l,r)>=target){
                min=Math.min(min,r-l+1);
                l++;
            }
        }

        return sum(nums,0, nums.length-1)<target?0:min;
    }

    private int sum(int[] nums,int l,int r){
        int sum=0;
        while(l<=r){
            sum=sum+nums[l];
            l++;
        }
        return sum;
    }
}
```
##### [438\. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)
```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        //记录p的所有字母及其个数
        char[] need = new char[26];
        for (int i = 0; i < p.length(); i++) {
            need[p.charAt(i) - 'a']++;
        }
        //start和end分别控制窗口的前端和后端
        int start = 0, end = 0;
        char[] window = new char[26];
        List<Integer> ans = new ArrayList<Integer>();
        while (end < s.length()) {
            window[s.charAt(end) - 'a']++; //加入窗口数据
            while (end - start + 1 == p.length()) { //维护一个长度为p.length()的窗口，并更新答案
                if (Arrays.equals(window, need)) ans.add(start);
                window[s.charAt(start) - 'a']--;
                start++;
            }
            end++;
        }
        return ans;
    }
}

```
 ##### [1004. 最大连续1的个数 III](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)


#### 中心扩展
##### [647.回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

```java
class Solution6472 {
    public int countSubstrings(String s) {
        // 中心扩展法
        int ans = 0;
        for (int center = 0; center < 2 * s.length() - 1; center++) {
            // left和right指针和中心点的关系是？
            // 首先是left，有一个很明显的2倍关系的存在，其次是right，可能和left指向同一个（偶数时），也可能往后移动一个（奇数）
            // 大致的关系出来了，可以选择带两个特殊例子进去看看是否满足。
            int left = center / 2;
            int right = left + center % 2;

            while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
                ans++;
                left--;
                right++;
            }
        }
        return ans;
    }
}
```

#### 位运算

#### [338\. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)
```java
class Solution {
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        for (int i = 0; i <= num; i++) {
            bits[i] = countOnes(i);
        }
        return bits;
    }

    public int countOnes(int x) {
        int ones = 0;
        while (x > 0) {
            x &= (x - 1);
            ones++;
        }
        return ones;
    }

}

```
##### [461\. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)
```java
class Solution {
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }
}
```

#### 标记
##### [448\. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)
```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result=new ArrayList<>();
        int n=nums.length;
        for(int num:nums){
            int x=(num-1)%n;
            nums[x]=nums[x]+n;
        }

        for(int i=0;i<n;i++){
            if(nums[i]<=n){
                result.add(i+1);
            }
        }
        return result;

    }
}

```


#### 先排序 
##### [406\. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)
```java
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        if(people.length==0) return new int[0][0];
        Arrays.sort(people,(o1, o2) ->{return o1[0]==o2[0]? o1[1]-o2[1]:o2[0]-o1[0];});

        List<int[]> queue = new ArrayList<>();
        for(int[] p:people){
            queue.add(p[1],p);
        }
        return queue.toArray(new int[0][]);
    }
}
```


#### 并查集

##### [399\. 除法求值](https://leetcode-cn.com/problems/evaluate-division/)
```
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {

    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int equationsSize = equations.size();

        UnionFind unionFind = new UnionFind(2 * equationsSize);
        // 第 1 步：预处理，将变量的值与 id 进行映射，使得并查集的底层使用数组实现，方便编码
        Map<String, Integer> hashMap = new HashMap<>(2 * equationsSize);
        int id = 0;
        for (int i = 0; i < equationsSize; i++) {
            List<String> equation = equations.get(i);
            String var1 = equation.get(0);
            String var2 = equation.get(1);

            if (!hashMap.containsKey(var1)) {
                hashMap.put(var1, id);
                id++;
            }
            if (!hashMap.containsKey(var2)) {
                hashMap.put(var2, id);
                id++;
            }
            unionFind.union(hashMap.get(var1), hashMap.get(var2), values[i]);
        }

        // 第 2 步：做查询
        int queriesSize = queries.size();
        double[] res = new double[queriesSize];
        for (int i = 0; i < queriesSize; i++) {
            String var1 = queries.get(i).get(0);
            String var2 = queries.get(i).get(1);

            Integer id1 = hashMap.get(var1);
            Integer id2 = hashMap.get(var2);

            if (id1 == null || id2 == null) {
                res[i] = -1.0d;
            } else {
                res[i] = unionFind.isConnected(id1, id2);
            }
        }
        return res;
    }

    private class UnionFind {

        private int[] parent;

        /**
         * 指向的父结点的权值
         */
        private double[] weight;


        public UnionFind(int n) {
            this.parent = new int[n];
            this.weight = new double[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                weight[i] = 1.0d;
            }
        }

        public void union(int x, int y, double value) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return;
            }

            parent[rootX] = rootY;
              // 关系式的推导请见「参考代码」下方的示意图
            weight[rootX] = weight[y] * value / weight[x];
        }

        /**
         * 路径压缩
         *
         * @param x
         * @return 根结点的 id
         */
        public int find(int x) {
            if (x != parent[x]) {
                int origin = parent[x];
                parent[x] = find(parent[x]);
                weight[x] *= weight[origin];
            }
            return parent[x];
        }

        public double isConnected(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) {
                return weight[x] / weight[y];
            } else {
                return -1.0d;
            }
        }
    }
}
```


## 多线程

- Semaphore
- CountDownLatch
- CyclicBarrier
- ReentrantLock
- synchronized
- BlockingQueue