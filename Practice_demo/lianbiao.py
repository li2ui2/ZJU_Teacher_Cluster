# ListNode只定义了__init__这个函数，那这个类的实例化对象只能表示一个节点
# 它虽然具有初始节点值，也有.next这个定义，但没有接下来其他类函数去定义节点关系
# 那它就只能表示一个节点。
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


#Solution类中的函数分别用递归和非递归的方法合并两张有序链表
class Solution:
    def mergeTwoLists1(self, l1, l2):
        """
        该函数用递归方法
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1==None and l2==None:
            return None
        if l1==None:
            return l2
        if l2==None:
            return l1
        if l1.val<=l2.val:
            l1.next=self.mergeTwoLists1(l1.next,l2)
            return l1
        else:
            l2.next=self.mergeTwoLists1(l1,l2.next)
            return l2
    def mergeTwoLists2(self, l1, l2):
        """
        该函数用非递归方法
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 is None and l2 is None:
            return None
        new_list = ListNode(0)
        pre = new_list
        while l1 is not None and l2 is not None:
            if l1.val < l2.val:
                pre.next = l1
                l1 = l1.next
            else:
                pre.next = l2
                l2 = l2.next
            pre = pre.next
        if l1 is not None:
            pre.next = l1
        else:
            pre.next = l2
        return new_list.next

#有序链表l1的定义
head1 = ListNode(2)
n1 = ListNode(3)
n2 = ListNode(4)
n3 = ListNode(9)
head1.next = n1
n1.next = n2
n2.next = n3

#有序链表l2的定义
head2 = ListNode(3)
m1 = ListNode(5)
m2 = ListNode(8)
m3 = ListNode(10)
head2.next = m1
m1.next = m2
m2.next = m3


s = Solution()
#递归调用合并两张链表
res = s.mergeTwoLists1(head1,head2)
#也可以非递归调用
#res = s.mergeTwoLists2(head1,head2)
while res:
    print(res.val)
    res = res.next