
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def _addHelper(self, l1, l2, head, curr, carry_over):
        if not l1 and not l2:
            if carry_over == 1:
                curr.next = ListNode(1)
            return head

        if not l1:
            val = l2.val + carry_over
            l2 = l2.next
        elif not l2:
            val = l1.val + carry_over
            l1 = l1.next
        else:
            val = l1.val + l2.val + carry_over
            l2 = l2.next
            l1 = l1.next

        next_node = ListNode(val % 10)

        if not head:
            head = next_node
            curr = head
        else:
            curr.next = next_node
            curr = next_node

        return self._addHelper(l1, l2, head, curr, val // 10)

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        return self._addHelper(l1, l2, None, None, 0)