# leetcode-daytime
leetcode刷题日志

## nSum问题

> 那一天的数组，排序起来；那一天的元素，组合起来；那一天的指针，移动起来
>
> 那一天的算法，跑不起来；那一天的思路，说不出口；那一天的面试，成了一坨

- 返回和为target的两个元素的下标
- 返回和为0的所有三元组
- 返回和与target最接近的三元组的和
- 返回和为target的所有四元组

### 2Sum

> 在狗血的人世间里，耗尽一生去找另一半

思路

> [1.两数之和](https://leetcode.cn/problems/two-sum)
>
> 遍历数组，确定数组中是否存在另外一个数，和当前遍历到的数相加之后，得到的和与target相等

解题过程

> 1. 创建哈希map，用于存储另外一个数和当前遍历到的数的下标的映射关系 
> 2. 遍历数组，检查数组中是否存在另外一个数和当前遍历到的数相加之和等于target

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(n) 

````go
func twoSum(arr []int, target int) []int {
	cache := make(map[int]int)
	for i := 0; i < len(arr); i++ {
		if other, ok := cache[arr[i]]; ok {
			return []int{other, i}
		} else {
			cache[target-arr[i]] = i
		}
	}
	return nil
}
````



### 3Sum

> 爱情三角理论：亲密，激情，承诺缺一不可

排序+双指针 解决三数之和

双指针

排序

> Problem: [15. 三数之和](https://leetcode.cn/problems/3sum/description/)﻿﻿

思路

> 先对数组进行排序，在外层遍历数组，先确定一个nums[i]，剩下只需要保证nums[left]+nums[right]==-nums[i]即可找到所求的三元组

解题过程

1. 排序：这步不必多说，一般都可以调用java或者是golang现成的排序方法，如果不给用，就可以手写一个快排来给这个整数数组排一下序，时间复杂度为O(nlogN)
2. 在剩余的元素中使用双指针法：一个left，一个right，先确定初始值，然后就依据nums[i]+nums[left]+nums[right]和0的比较关系，来确定执行什么操作：

- 若nums[i]+nums[left]+nums[right]==0：找到了一个符合要求的三元组，添加到结果数组中，然后为了保证下一次循环不会找到重复的三元组，需要将left和right都移动到最近的一个不相同的元素处（这一步去重操作容易写错），具体做法就是先通过循环+nums[left]==nums[left+1]条件将left移动到最后一个重复元素位置处，然后再将left向后移动一格，就可以去重成功（不能够使用nums[left-1]==nums[left]条件，因为这样会出现数组越界的问题）
- 若nums[i]+nums[left]+nums[right]<0：说明整体偏小，故将left右移来增大整体的的和
- 若nums[i]+nums[left]+nums[right]>0：说明整体偏大，故将right左移来缩小整体的和

复杂度

- 时间复杂度: O(nlogN)+O(n^2)
- 空间复杂度: O(1)

Code

````go
func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	var res [][]int
	for i := 0; i < len(nums); i++ {
		for i > 0 && i<len(nums) && nums[i-1] == nums[i] {
			i++
		}
		left := i + 1
		right := len(nums) - 1
		for left < right {
			if nums[i]+nums[left]+nums[right] == 0 {
				res = append(res, []int{nums[i], nums[left], nums[right]})
				for left < right && nums[left+1] == nums[left] {
					left++
				}
				for left < right && nums[right-1] == nums[right] {
					right--
				}
				left++
				right--
			} else if nums[i]+nums[left]+nums[right] < 0 {
				left++
			} else if nums[i]+nums[left]+nums[right] > 0 {
				right--
			}
		}
	}
	return res
}
````



### 4Sum

> 一生二，二生三，三生四，四归混沌

> Problem: [18. 四数之和](https://leetcode.cn/problems/4sum/description/)

思路

> 先对数组进行排序，之后通过两层循环确定四元组的前两个数字nums[i]和nums[j]，接着剩余的两个数字可以通过双指针法进行确定

解题过程

1. 排序：简单，不必多说
2. 两层for循环：先确定两个数字，将四数之和问题转换成两数之和问题
3. 双指针法：使用对撞双指针left和right，这里就只有去重操作需要注意一下，特别是要理解这样写的原因是什么，注意去重的细节不要写错（特别留意开始执行去重操作的时机）

复杂度

- 时间复杂度: O(nlogN) + O(n^3) 
- 空间复杂度: O(1)

Code

````go
func fourSum(nums []int, target int) [][]int {
	sort.Ints(nums)
	var res [][]int
	for i := 0; i < len(nums); i++ {
		for i > 0 && i < len(nums) && nums[i-1] == nums[i] {
			i++
		}
		for j := i + 1; j < len(nums); j++ {
			for j > i+1 && j < len(nums) && nums[j-1] == nums[j] {
				j++
			}
			left := j + 1
			right := len(nums) - 1
			for left < right {
				if nums[i]+nums[j]+nums[left]+nums[right] == target {
					res = append(res, []int{nums[i], nums[j], nums[left], nums[right]})
					for left < right && nums[left] == nums[left+1] {
						left++
					}
					for left < right && nums[right-1] == nums[right] {
						right--
					}
					left++
					right--
				} else if nums[i]+nums[j]+nums[left]+nums[right] < target {
					left++
				} else if nums[i]+nums[j]+nums[left]+nums[right] > target {
					right--
				}
			}
		}
	}
	return res
}
````



## 链表问题

> 芳林新叶催陈叶，流水前波让后波。

### 删除单链表的倒数第N个结点

> 在好友列表的倒数第N个人，在聊天界面的倒数第N行，在我和你说的倒数第N句话：删了吧，就此别过

> Problem: [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/)

思路

> 快慢双指针，快指针位于慢指针前面第N+1个节点，也就是说当快指针指向末尾nil处时，慢指针所处的位置无疑就是待删除节点的前驱节点

解题过程

1. dummy结点：用了虚结点之后，那么就可以极大程度上避免链表的边界判断以及空指针的问题，这个是必用的，不必多言
2. 先移动快指针：前面说了，快指针充当的作用就是作为一个标记物，专门用来标记末尾，然后相对这个标记物，往前n个位置就是待删除的结点，往前n+1个位置就是待删除结点的前驱节点
3. 快慢指针同时移动：直到快指针真正到达了nil处
4. 删除结点：让前驱节点指向后继节点的下一个结点即可，这里要小心那该死的空指针异常，也就是说必须要先判定前驱节点和这个后继节点是不是都不为nil

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(1) 

Code

````go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	second := dummy
	first := head
	for n > 0 && first != nil {
		first = first.Next
		n--
	}
	for first != nil {
		second = second.Next
		first = first.Next
	}
	removeNode(second, second.Next)
	return dummy.Next
}

func removeNode(prev *ListNode, node *ListNode) {
	if prev == nil || node == nil {
		return
	}
	prev.Next = node.Next
}
````

### 合并两个有序链表

> 井水不犯河水，大雨冲了龙王庙

> Problem: [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/)

思路

> 双指针法

解题过程

> 由于两个链表都是有序的，所以可以通过比较两个链表中的元素，来决定新链表中的元素顺序，需要注意的是，当两个链表的长度不一样时，在比较结束时，会出现有一个链表还有剩余元素的情况，这个时候，只需要将剩余元素全部添加到新链表末尾即可

复杂度

- 时间复杂度: O(n)
- 空间复杂度: O(n)

Code

````go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(h1 *ListNode, h2 *ListNode) *ListNode {
	p1 := h1
	p2 := h2

	dummy := &ListNode{}
	p3 := dummy
	for p1 != nil && p2 != nil {
		if p1.Val < p2.Val {
			p3.Next = p1
			p1 = p1.Next
			p3 = p3.Next
		} else if p1.Val >= p2.Val {
			p3.Next = p2
			p2 = p2.Next
			p3 = p3.Next
		}
	}
	if p1 != nil {
		p3.Next = p1
	}
	if p2 != nil {
		p3.Next = p2
	}

	return dummy.Next
}
````

### 合并k个有序链表

> 网络上，我分治法随手拈来，现实里，我选择先合并两个链表，剩下的明天再说（前提是今天不是ddl）

思路

> [18.合并k个有序链表](https://leetcode.cn/problems/4sum/description)
>
> 这么简单也配困难等级？只需要将这k个链表两两合并即可，然后两个链表合并还是用回双指针老套路就能解决

解题过程

两两合并不秒了吗？

复杂度

- 时间复杂度: O(n^2) 
- 空间复杂度: O(n)

Code

````go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeKLists(lists []*ListNode) *ListNode {
	if len(lists) == 0 {
		return nil
	} else if len(lists) == 1 {
		return lists[0]
	}
	var tmp *ListNode
	tmp = mergeTwoList(lists[0], lists[1])
	for i := 2; i < len(lists); i++ {
		tmp = mergeTwoList(tmp, lists[i])
	}
	return tmp
}

func mergeTwoList(h1 *ListNode, h2 *ListNode) *ListNode {
	p1 := h1
	p2 := h2

	dummy := &ListNode{}
	p3 := dummy
	for p1 != nil && p2 != nil {
		if p1.Val < p2.Val {
			p3.Next = p1
			p1 = p1.Next
			p3 = p3.Next
		} else if p1.Val >= p2.Val {
			p3.Next = p2
			p2 = p2.Next
			p3 = p3.Next
		}
	}
	if p1 != nil {
		p3.Next = p1
	}
	if p2 != nil {
		p3.Next = p2
	}

	return dummy.Next
}
````

### 两两交换链表中的结点

> "你打伞打了那么久，也应该换我用一下吧"
>
> "可是你已经淋湿了，如果把伞换给你，那我们都会淋湿的，这样毫无意义，你说对吧"

> Problem: [24. 两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/description/)

思路

> 双指针法

解题过程

> 每次只需要交换两个指针指向的结点即可，并且需要额外维护一个前驱结点才能实现交换

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(1)

Code

````go
func swapPairs(h *ListNode) *ListNode {
	if h == nil {
		return nil
	}
	dummy := &ListNode{}
	dummy.Next = h
	first := h.Next
	second := h
	prev := dummy
	for first != nil && second != nil {
		swap(prev, second, first)
		prev = second
		second = second.Next
		if second != nil {
			first = second.Next
		}
	}
	return dummy.Next
}

func swap(prev *ListNode, l1 *ListNode, l2 *ListNode) {
	prev.Next = l2
	l1.Next = l2.Next
	l2.Next = l1
}
````

### 环形链表

> 有缘（园）才相遇

> Problem: [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/description/)

思路

> 设置快慢两个指针，快指针每次前进两个单位，慢指针每次前进一个单位，快指针相对于慢指针的速度是每次一个单位，所以，二者的间距会逐步拉大，假设当前链表是一个环形链表，从相对速度来看，慢指针和快指针在圆上做同向运动，慢指针相当于不动，而快指针相当于每次前进一个单位，很明显，快指针是一定能够遍历圆上的所有结点，当然也包括慢指针所在的那个结点，所以如果链表含有环，那么二者一定能够相遇

解题过程

> 设置快慢两个指针，快指针每次前进两个单位，慢指针每次前进一个单位

复杂度

- 时间复杂度: O(n)
- 空间复杂度: O(1)  

Code

````go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func hasCycle(h *ListNode) bool {
	if h == nil {
		return false
	}
	second := h
	first := h.Next
	for first != nil && first.Next != nil {
		if first == second {
			return true
		}
		first = first.Next.Next
		second = second.Next
	}
	return false
}
````

### 有序链表去重

给一个有序链表去掉重复元素

todo

### 链表排序

todo

将一个无序链表排序成升序链表

二分+合并排序

### 有序链表转换成二叉平衡树

todo

中序遍历

## 字符串问题

> 细线牵春色，幽窗弄晓愁

### 无重复字符的最长子串

> Problem: [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)

思路

> 滑动窗口法

解题过程

1. 先确定左指针
2. 维护滑动窗口左边界：要将左指针左边的字符从哈希表中删除
3. 移动右指针，直到遇到的字符已经在滑动窗口中存在了
4. 计算滑动窗口中的字符数量，并且与之前的最长字符数量进行比较择优

细节部分

1. golang字符串采用utf-8编码，golang没有专门的字符类型，而是使用整数类型来存储字符的unicode码值，其中ACSII表范围以内的（即码值255以下的，一般是英文字母还有数字之类的）采用byte类型，超出ACSII表范围的（一般是中文，日文等等）采用int类型
2. golang map类型如果key不存在那么不会返回nil而是返回相应类型的零值，比如int类型就是返回0

复杂度

- 时间复杂度:  O(n^2) 
- 空间复杂度: O(n)

Code

````go
func lengthOfLongestSubstring(s string) int {
	window := map[byte]int{}
	right := -1
	maxLen := 0
	n := len(s)
	for i := 0; i < n; i++ {
		if i != 0 {
			delete(window, s[i-1])
		}
		for right+1 < n && window[s[right+1]] == 0 {
			right++
			window[s[right]]++
		}
		curLen := right - i + 1
		if curLen > maxLen {
			maxLen = curLen
		}
	}
	return maxLen
}
````

### 验证回文串

> Problem: [125. 验证回文串](https://leetcode.cn/problems/valid-palindrome/description/)

思路

> 对撞双指针

解题过程

> 分为两个部分，使用两个相向的指针，同时移动，同时比较两个指针所指位置的字符是否相等，如果不相等直接返回false，相等就同时移动，直到两个指针相遇都还是相等的话，就可以确认这是一个回文串了

复杂度

- 时间复杂度: O(n) O(n) O(n)
- 空间复杂度: O(n) O(n) O(n)

Code

````go
func isPalindrome(s string) bool {
	bytes := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		if (s[i] >= 'a' && s[i] <= 'z') || (s[i] >= '0' && s[i] <= '9') {
			bytes = append(bytes, s[i])
		} else if s[i] >= 'A' && s[i] <= 'Z' {
			bytes = append(bytes, s[i]-'A'+'a')
		}
	}
	s = string(bytes)
	fmt.Println(s)
	left := 0
	right := len(s) - 1
	for left < right {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}
	return true
}
````

### 有效的括号

> Problem: [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/description/)

思路

> 栈

解题过程

> 遇到左半边的括号就压进栈内，遇到右半边的括号就弹出栈内元素，并且确定是否匹配

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(n) 

Code

````go
func isValid(s string) bool {
	stack := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		if s[i] == '{' || s[i] == '(' || s[i] == '[' {
			stack = append(stack, s[i])
		} else if s[i] == '}' {
			if len(stack) == 0 {
				return false
			}
			last := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if last != '{' {
				return false
			}
		} else if s[i] == ']' {
			if len(stack) == 0 {
				return false
			}
			last := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if last != '[' {
				return false
			}
		} else if s[i] == ')' {
			if len(stack) == 0 {
				return false
			}
			last := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if last != '(' {
				return false
			}
		}

	}
	if len(stack) != 0 {
		return false
	}
	return true
}
````

### 最长有效括号

> Problem: [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/description/)

思路

> 动态规划

解题过程

![img](https://pic.leetcode.cn/1752408987-XQnHfU-image.png)

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(n) 

Code

````go
func longestValidParentheses(s string) int {
	dp := make([]int, len(s), len(s))
	maxLen := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			dp[i] = 0
		} else if s[i] == ')' {
			if i > 0 && s[i-1] == '(' {
				if i > 1 {
					dp[i] = dp[i-2] + 2
				} else {
					dp[i] = 2
				}
			} else if i > 0 && s[i-1] == ')' {
				if i >= 1+dp[i-1] && s[i-dp[i-1]-1] == '(' {
					if i >= dp[i-1]+2 {
						dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]
					} else {
						dp[i] = dp[i-1] + 2
					}
				}
			}
		}
		maxLen = maxInt(maxLen, dp[i])
	}
	return maxLen
}

func maxInt(x int, y int) int {
	if x > y {
		return x
	} else {
		return y
	}
}
````

## DFS

### 组合

> Problem: [77. 组合](https://leetcode.cn/problems/combinations/description/)

思路

> dfs

解题过程

> 剪枝+选择或不选择+深度递归遍历

复杂度

- 时间复杂度: O(n)
- 空间复杂度: O(n)

Code

````go
var temp []int
var ans [][]int

func combine(n int, k int) [][]int {
	temp = temp[0:0]
	ans = ans[0:0]
	dfs(n, 1, k)
	return ans
}

func dfs(n int, cur int, k int) {
	if k-len(temp) > n-cur+1 {
		return
	}
	if len(temp) == k {
		copySlice := make([]int, len(temp))
		copy(copySlice, temp)
		ans = append(ans, copySlice)
		return
	}

	temp = append(temp, cur)
	dfs(n, cur+1, k)
	temp = temp[:len(temp)-1]
	dfs(n, cur+1, k)
}
````

### 二叉树的层序遍历

> Problem: [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/)

思路

> DFS

解题过程

> 从根节点开始，使用深度优先搜索方法，将当前遍历到的结点，添加到对应层级的数组中就行了，递归的写法是比非迭代+队列的写法要简单很多，而且不容易出bug

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(n)

Code

````go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
var ans [][]int

func levelOrder(root *TreeNode) [][]int {
    ans = ans[0:0]
	if root == nil {
		return ans
	}
	dfs(0, root)
	return ans
}

func dfs(level int, root *TreeNode) {
	if len(ans) == level {
		ans = append(ans, []int{})
	}
	ans[level] = append(ans[level], root.Val)
	if root.Left != nil {
		dfs(level+1, root.Left)
	}
	if root.Right != nil {
		dfs(level+1, root.Right)
	}
}
````



## 双指针

### 验证回文串

### 三数之和

### 盛最多水的容器

> Problem: [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/)

思路

> 对撞双指针

解题过程

> 可能会想到使用双指针，但是这是个无序数组，很难想到应该如何移动这两个指针，才能保证找到全局最优解。 所以，先来考虑使用暴力法来解决，如果使用暴力法的话，是个人都知道要用两层for循环穷举所有可能情况，然后选出最大的即可，但是这样做的时间复杂度太高了，所以考虑能否少考虑一些情况。通过观察可以知道，如果保持高度不变的情况下，缩小宽度的话，面积肯定是缩小的，所以这些情况都可以排除掉不用去计算了，也就是不能够保持高度不变，而是要通过移动来让高度也发生变化，这样的话才有可能在之后的移动中找到更大的面积。

复杂度

- 时间复杂度: O(n)
- 空间复杂度: O(1) 

Code

````go
func maxArea(height []int) int {
	maxA := 0
	left := 0
	right := len(height) - 1
	for left < right {
		cur := (right - left) * minInt(height[left], height[right])
		maxA = maxInt(maxA, cur)
		if height[left] < height[right] {
			left++
		} else {
			right--
		}
	}
	return maxA
}

func maxInt(x int, y int) int {
	if x > y {
		return x
	} else {
		return y
	}
}

func minInt(x int, y int) int {
	if x < y {
		return x
	} else {
		return y
	}
}
````



### 判断子序列

### 两数之和Ⅱ输入有序数组

## 区间

### 合并区间

> Problem: [56. 合并区间](https://leetcode.cn/problems/merge-intervals/description/)

思路

> 先将原来的区间数组按照左边界进行排序

解题过程

1. 预处理：先将原来的区间数组按照左边界进行排序
2. 遍历区间数组：比较区间边界，以此来确定，是否要进行区间合并

复杂度

- 时间复杂度: O(nlogN)+O(n)  O(nlogN) +O(n) O(nlogN)+O(n)
- 空间复杂度: O(n) O(n) O(n)

Code

````go
func merge(intervals [][]int) [][]int {
	if len(intervals) < 2 {
		return intervals
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	var ans [][]int
	for i := 0; i < len(intervals); i++ {
		if len(ans) == 0 {
			ans = append(ans, intervals[i])
		}
		last := ans[len(ans)-1]
		if last[1] >= intervals[i][0] {
			ans[len(ans)-1][1] = maxInt(last[1], intervals[i][1])
		} else {
			ans = append(ans, intervals[i])
		}
	}
	return ans
}

func maxInt(x int, y int) int {
	if x > y {
		return x
	} else {
		return y
	}
}
````

### 插入区间

## 一维动规

### 爬楼梯

> Problem: [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/)

思路

> 一维动规

解题过程

> 从第0级开始，运用加法原理可以知道，到达第i级的方法有两个： 1. 先到达第i-1级，然后再走1级，就可以到达第i级 2. 先到达第i-2级，然后再走2级，就可以到达第i级 由加法原理就可以知道，到达第i级总的方案数量就是到达第i-1级和第i-2级的方案数量之和 边界条件： 为了保证级数为非负，就要先确定第0级和第1级的方案数量，第0级有一种走法，就是从第0级到第0级，第1级也是一种走法，就是从第0级到第1级

复杂度

- 时间复杂度: O(n)
- 空间复杂度: O(n) 

Code

````go
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i < len(dp); i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
````

### 打家劫舍

> Problem: [198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/)

思路

> 一维动规

解题过程

> 我们先来思考一下，dp[i]应该代表什么含义才有利于我们后面的思考和状态转移呢，毫无疑问，必须要让dp[i]和nums[i]扯上关系，那么就可以让dp[i]表示以nums[i]结尾的可以偷盗的最大金额，因为只有先确定了末尾元素，后面状态转移的时候，才有可以比较的标志物。接着我们再来思考状态转移的过程是怎样，由题目可以确定dp[i]由于是以nums[i]结尾的，所以必定包括了nums[i]，并且不包括nums[i-1]，并且可以从0到i-2任意一个状态转移到i，又因为要让总和最大，所以上一个状态必定是所有可选状态中的最大值，那么我们就可以在程序中维护dp[0]到dp[i-2]的最大值，那么dp[i]=max{0,1,...,i-2}+nums[i]

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(n) 

Code

````go
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	} else if len(nums) == 1 {
		return nums[0]
	}
	dp := make([]int, len(nums))
	maxNum := 0
	ans := 0
	dp[0] = nums[0]
	dp[1] = nums[1]
	maxNum = dp[0]
	ans = maxInt(dp[0], dp[1])
	for i := 2; i < len(nums); i++ {
		dp[i] = maxNum + nums[i]
		ans = maxInt(ans, dp[i])
		maxNum = maxInt(maxNum, dp[i-1])
	}
	return ans
}

func maxInt(x int, y int) int {
	if x > y {
		return x
	} else {
		return y
	}
}
````

### 打家劫舍Ⅱ

> Problem: [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/description/)

思路

> 一维动规

解题过程

> 这个和1的区别在于，不可以首位相连，那就在1的基础上进行改造，分为两种情况，取其中的最大值即可： 1. 偷了第1家，就不能偷最后一家 2. 偷了最后一家，就不能偷第一家

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(n)

Code

````go
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}else if len(nums) == 1 {
		return nums[0]
	}else if len(nums) == 2 {
		return maxInt(nums[0],nums[1])
	}
	left := robRange(nums, 0, len(nums)-2)
	right := robRange(nums, 1, len(nums)-1)
	return maxInt(left, right)
}

func robRange(nums []int, start int, end int) int {
	dp := make([]int, len(nums))
	dp[start] = nums[start]
	dp[start+1] = maxInt(nums[start], nums[start+1])
	for i := start + 2; i <= end; i++ {
		dp[i] = maxInt(nums[i]+dp[i-2], dp[i-1])
	}
	return dp[end]
}

func maxInt(x int, y int) int {
	if x > y {
		return x
	} else {
		return y
	}
}
````

### 硬币问题

> Problem: [322. 零钱兑换](https://leetcode.cn/problems/coin-change/description/)

思路

> 一维动态规划

解题过程

> dp[i]表示如果要使用coins[]中的面值，凑成金额i，所需要的最少的硬币数量 dp[0]=0意思是凑成金额0元，需要0个硬币，这是可以确定的，也是最初的状态 dp[i]=min{dp[i]，dp[i-coins[j]]}表示coins[]提供的面额都试一遍，选择可行的并且硬币数量最少的那个

复杂度

- 时间复杂度: O(n2)
- 空间复杂度: O(n) 

Code

````go
func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		dp[i] = math.MaxInt32
	}
	for i := 1; i <= amount; i++ {
		for j := 0; j < len(coins); j++ {
			if i >= coins[j] && dp[i-coins[j]] != math.MaxInt32 {
				dp[i] = minInt(dp[i], 1+dp[i-coins[j]])
			}
		}
	}
	if dp[amount] == math.MaxInt32 {
		return -1
	}
	return dp[amount]
}

func minInt(x int, y int) int {
	if x < y {
		return x
	} else {
		return y
	}
}
````

## 二维动规

### 买卖股票的最佳时机

> Problem: [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/)

思路

> 二维动规

解题过程

> dp[i][0]：表示在第i天，状态是手上没有持有股票，可能是之前就没有买，或者是之前买的在今天才卖掉 dp[i][1]：表示在第i天，状态时手上持有股票，可能是之前买的，或者是今天买的

复杂度

- 时间复杂度: O(n) 
- 空间复杂度: O(n)

Code

````go
func maxProfit(prices []int) int {
	dp := make([][2]int, len(prices))
	dp[0][0] = 0
	dp[0][1] = -prices[0]
	for i := 1; i < len(prices); i++ {
		dp[i][1] = maxInt(dp[i-1][1], -prices[i])
		dp[i][0] = maxInt(dp[i-1][0], dp[i-1][1]+prices[i])
	}
	return dp[len(prices)-1][0]
}

func maxInt(x int, y int) int {
	if x > y {
		return x
	} else {
		return y
	}
}
````



[]: 
