/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef List_h
#define List_h

namespace bmalloc {

template<typename T>
struct ListNode {
    ListNode<T>* prev { nullptr };
    ListNode<T>* next { nullptr };
};

template<typename T>
class List {
    static_assert(std::is_trivially_destructible<T>::value, "List must have a trivial destructor.");

    struct iterator {
        T* operator*() { return static_cast<T*>(m_node); }
        T* operator->() { return static_cast<T*>(m_node); }

        bool operator!=(const iterator& other) { return m_node != other.m_node; }

        iterator& operator++()
        {
            m_node = m_node->next;
            return *this;
        }
        
        ListNode<T>* m_node { nullptr };
    };

public:
    List() { }

    bool isEmpty() { return m_root.next == &m_root; }

    T* head() { return static_cast<T*>(m_root.next); }
    T* tail() { return static_cast<T*>(m_root.prev); }
    
    iterator begin() { return iterator { m_root.next }; }
    iterator end() { return iterator { &m_root }; }

    void push(T* node)
    {
        ListNode<T>* it = tail();
        insertAfter(it, node);
    }

    void pushFront(T* node)
    {
        ListNode<T>* it = &m_root;
        insertAfter(it, node);
    }

    T* pop()
    {
        ListNode<T>* result = tail();
        remove(result);
        return static_cast<T*>(result);
    }

    T* popFront()
    {
        ListNode<T>* result = head();
        remove(result);
        return static_cast<T*>(result);
    }

    static void insertAfter(ListNode<T>* it, ListNode<T>* node)
    {
        ListNode<T>* prev = it;
        ListNode<T>* next = it->next;

        node->next = next;
        next->prev = node;

        node->prev = prev;
        prev->next = node;
    }

    static void remove(ListNode<T>* node)
    {
        ListNode<T>* next = node->next;
        ListNode<T>* prev = node->prev;

        next->prev = prev;
        prev->next = next;
        
        node->prev = nullptr;
        node->next = nullptr;
    }

private:
    ListNode<T> m_root { &m_root, &m_root };
};

} // namespace bmalloc

#endif // List_h
