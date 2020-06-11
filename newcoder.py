# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:14:01 2020

@author: 76754
"""

#1. 两个栈实现队列(Two Stacks to realize queue)

#using two stacks to achieve the reversed order.
#1. using pushlist to store the pushed element. using poplist to pop the first element in the pushlist(queue)
#2. if poplist has element, just pop the last element. if poplist has no element and push list has elements, 
#poping the elemetns from the pushlist and append them into the poplist.
class twoStackQueue:
    def __init__(self):
        self.pushlist=[]
        self.poplist=[]
    def push(self,new):
        self.pushlist.append(new)
    def pop(self):
        if len(self.poplist)>0:
            return self.poplist.pop()
        else:
            while len(self.pushlist)>0:
                self.poplist.append(self.pushlist.pop())
            return self.poplist.pop()
a=twoStackQueue()
a.push(1)
a.push(2)
a.pop()
#2. 二维有序数组中的查找(Search element in an ordered two-dim matrix)
            
#for each row, the element is stored in a ascending order
#select the lower-left element as the element to compare with
#3 possibilities: (1) target==array[m][n] find (2) target>array[m][n] let m=m-1 (3) target<array[m][n] let n=n+1
# set boundaries: m,n,find

def Find( target, array):
    m=len(array)-1
    n=0
    find=False
    while not find and m>=0 and n<=len(array[0])-1:
        if array[m][n]==target:
            find=True
        elif array[m][n]<target:
            n+=1
        else:
            m-=1
    return find

def Find_binarysearch(target, array):
    find=False
    for rows in array:
        left=0
        right=len(rows)-1
        while left<=right:
            tmp=(left+right)//2
            if rows[tmp]==target:
                return True
            elif rows[tmp]>target:
                right=tmp-1
            else:
                left=tmp+1
    return find 

#3. 输入链表，按列表从尾到头返回一个list(returning a linnked list with inversed order)
class ListNode:
    def __init__(self,data):
        self.val=data
        self.next=None
class Solution:
    def printListFromTailToHead(self,listNode):
        stack=[]
        new=[]
        if not listNode:
            return new
        while listNode:
            stack.append(listNode.val)
            listNode=listNode.next
        while stack:
            new.append(stack.pop())
        return new
#4. 重建二叉树：输入前序遍历序列与中序遍历序列，返回重建的二叉树（restore binary tree） 
#（1）递归构建树、（2）前序遍历第一个元素为根节点 （3）中序遍历根节点两边分别为左右两枝


# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre or not tin:
            return None
        root=TreeNode(pre.pop(0))
        index=tin.index(root.val)
        root.left=self.reConstructBinaryTree(pre,tin[:index])
        root.right=self.reConstructBinaryTree(pre,tin[index+1:])
        return root
store=[0]*(n+1)
store[1]=1
for i in range(2,len(store)):
    store[i]=store[i-1]+store[i-2]
# 5. 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
#例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
#NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
    
#（1）二分查找变体，同样使用二分查找
#（2）若中间的数小于最右的数，则最右侧数变为中间数，不可向下由于该数可能恰好为最小值
#（3）若中间的数大于最右的数，则概数位于左侧，则左侧树变为中间值+1
#（4）若中间数等于最右的树，则为2,3,1,1情况，则最右侧数逐渐减少1进行比较
#（5）返回最左侧的数
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        left=0
        right=len(rotateArray)-1
        if len(rotateArray)==0:
            return 0
        while left<right:
            tmp=(left+right)//2
            if rotateArray[tmp]<rotateArray[right]:
                right=tmp
            elif rotateArray[tmp]>rotateArray[right]:
                left=tmp+1
            else:
                right=right-1
        return rotateArray[left]
    
# 6. 斐波拉契
# 利用append
# 利用交换
#####注意 问题的场景，是否包含初始位置0，以及开始的一般情况
class Solution:
    def Fibonacci(self,n):
        first=0
        second=1
        if n==0:
            return  first
        elif n==1:
            return second
        else:
            for i in range(2,n+1):
                third=first+second
                first=second
                second=third
            return third
class Solution:
    def Fibonacci(self,n):
        # write code here 
        list1=[0,1]
        if n<=1:
            return list1[n]
        else:
            for i in range(2,n+1):
                list1.append(list1[i-1]+list1[i-2])
            return list1[n]
        
#7. 变态跳台阶
#一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
#每次都之前所有情况的求和，并且加上直接跳到当前位置的一次 f(n)=f(1)+...+f(n-1)+1
class Solution:
    def jumpFloorII(self, number):
        # write code here
        list1=[0,1,2]
        for i in range(3,number+1):
            list1.append(sum(list1[:i])+1)
        return list1[number]



#8. 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
#保证base和exponent不同时为0

#暴力求解
class Solution:
    def Power(self, base, exponent):
        # write code here
        result=1
        if base==0:
            return 0
        elif exponent ==0:
            return 1
        elif exponent<0:
            for _ in range(-exponent):
                result=result*base
            return 1/result
        else:
            for _ in range(exponent):
                result=result*base
            return result

# 快速幂
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base==0:
            return 0
        elif exponent==0:
            return 1
        else: 
            result=1
            expo=abs(exponent)
            while expo!=0:
                if expo&1==1:
                    result=base*result
                expo=expo>>1
                base=base*base
            return result if exponent>0 else 1/result
        
def fastExpMod(b, e, m):
    result = 1
    while e != 0:
        if (e&1) == 1:
            # ei = 1, then mul
            result = (result * b) % m
        e >>= 1
        # b, b^2, b^4, b^8, ... , b^(2^n)
        b = (b*b) % m
    return result



#9.输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
# 并保证奇数和奇数，偶数和偶数之间的相对位置不变。
    
# 新建list
class Solution:
    def reOrderArray(self, array):
        # write code here
        odd=[]
        even=[]
        for i in array:
            if i%2==1:
                odd.append(i)
            elif i%2==0:
                even.append(i)
        odd.extend(even)
        return odd
    
#冒泡排序
class Solution:
    def reOrderArray( array):
        # write code here
        flag=True
        while flag :
            flag=False
            for i in range(1,len(array)):
                if array[i-1]%2==0 and array[i]%2==1:
                    array[i-1],array[i]=array[i],array[i-1]
                    flag=True
        return array
# 10.输入一个链表，输出该链表中倒数第k个结点。
#栈
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if not head:
            return head
        
        stack=[]
        while head:
            stack.append(head)
            head=head.next
        if k>len(stack) or k<1:
            return
        return stack[-k]

#做一把尺子，然后共同移动
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if head==None or k<1:
            return 
        l1=l2=head
        for _ in range(k):
            if l1==None:
                return 
            else:
                l1=l1.next
        while l1:
            l1=l1.next
            l2=l2.next
        return l2
#11. 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        cur=ListNode(0)
        tmp=cur
        while pHead1 and pHead2:
            if pHead1.val<=pHead2.val:
                cur.next=pHead1
                pHead1=pHead1.next
            else:
                cur.next=pHead2
                pHead2=pHead2.next
            cur = cur.next
        if not pHead1:
            cur.next=pHead2
        elif not pHead2:
            cur.next=pHead1
        return tmp.next
    
#递归
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if pHead1 is None:
            return pHead2
        elif pHead2 is None:
            return pHead1
        if pHead1.val<pHead2.val:
            pHead1.next=self.Merge(pHead1.next, pHead2)
            return pHead1
        else:
            pHead2.next=self.Merge(pHead1, pHead2.next)
            return pHead2
        
#12. 输入一个链表，反转链表后，输出新链表的表头。
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        cur=pHead
        tmp=cur
        pre=None
        while cur!= None:
            tmp=cur.next
            cur.next=pre
            pre=cur
            cur=tmp
        return pre
    
#13.输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵：
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.   
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        result=[]
        while matrix:
            result=result+matrix.pop(0)
            if matrix and matrix[0]:
                for row in matrix:
                    result=result+[(row.pop())]
            if matrix:
                result=result+(matrix.pop())[::-1]
            if matrix and matrix[0]:
                for row in range(len(matrix)-1,-1,-1):
                    result=result+[matrix[row].pop(0)]
        return result
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        result = []
        while(matrix):
            result+=matrix.pop(0)
            if not matrix or not matrix[0]:
                break
            matrix = self.turn(matrix)
        return result
    def turn(self,matrix):
        num_r = len(matrix)
        num_c = len(matrix[0])
        newmat = []
        for i in range(num_c):
            newmat2 = []
            for j in range(num_r):
                newmat2.append(matrix[j][i])
            newmat.append(newmat2)
        newmat.reverse()
        return newmat


#14.输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
#（1）判断树种右相同的根节点（递归）
#(2)判断在有相同根节点的情况下，左右节点是否相同，到达某个叶节点停止
class Solution:
def HasSubtree(self, pRoot1, pRoot2):
    # write code here
    result=False
    if pRoot1!=None and pRoot2!=None:
        if pRoot1.val==pRoot2.val:
            result=self.DoesTree1haveTree2(pRoot1, pRoot2)
        if not result:
            result=self.HasSubtree(pRoot1.left, pRoot2)
        if not result:
            result=self.HasSubtree(pRoot1.right, pRoot2)
    return result

def DoesTree1haveTree2(self,pRoot1,pRoot2):
    if pRoot2 == None:
        return True
    if pRoot1 == None:
        return False
    if pRoot1.val != pRoot2.val:
        return False
    return self.DoesTree1haveTree2(pRoot1.left, pRoot2.left) and self.DoesTree1haveTree2(pRoot1.right, pRoot2.right)

#15. 定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
#注意：保证测试中不会当栈为空的时候，对栈调用pop()或者min()或者top()方法。
class Solution:
    def __init__(self):
        self.stack = []
        self.assist = []
         
    def push(self, node):
        min = self.min()
        if not min or node < min:
            self.assist.append(node)
        else:
            self.assist.append(min)
        self.stack.append(node)
         
    def pop(self):
        if self.stack:
            self.assist.pop()
            return self.stack.pop()
     
    def top(self):
        # write code here
        if self.stack:
            return self.stack[-1]
         
    def min(self):
        # write code here
        if self.assist:
            return self.assist[-1]
    
    
#16 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。 
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence or len(sequence)==0:
            return False
        root=sequence[len(sequence)-1]
        for i in range(len(sequence)):
            if sequence[i]>root:
                break
        for j in range(i,len(sequence)):
            if sequence[j]<root:
                return False
        left=True
        if i>0:
            left=self.VerifySquenceOfBST(sequence[:i])
        right=True
        if i<len(sequence)-1:
            right=self.VerifySquenceOfBST(sequence[i:-1])
        return left and right    
