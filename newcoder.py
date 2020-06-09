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
