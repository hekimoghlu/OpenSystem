/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#pragma once

#include <wtf/Bag.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/SentinelLinkedList.h>

namespace WTF {

// This is a collection that is meant to be used for building up lists in a certain order. It's
// not an efficient data structure for storing lists, but if you need to build a list by doing
// operations like insertBefore(existingValue, newValue), then this class is a good intermediate
// helper. Note that the type it operates on must be usable as a UncheckedKeyHashMap key.
template<typename T>
class OrderMaker {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(OrderMaker);
    
    struct Node : BasicRawSentinelNode<Node> {
        Node(SentinelTag)
        {
        }

        Node()
        {
        }

        T payload { };
    };
    
public:
    OrderMaker()
    {
    }

    void prepend(T value)
    {
        m_list.push(newNode(value));
    }

    void append(T value)
    {
        m_list.append(newNode(value));
    }

    void insertBefore(T existingValue, T newValue)
    {
        Node* node = m_map.get(existingValue);
        ASSERT(node);
        node->prepend(newNode(newValue));
    }
    
    void insertAfter(T existingValue, T newValue)
    {
        Node* node = m_map.get(existingValue);
        ASSERT(node);
        node->append(newNode(newValue));
    }

    class iterator {
    public:
        iterator()
        {
        }

        iterator(typename SentinelLinkedList<Node>::iterator iter)
            : m_iter(iter)
        {
        }

        const T& operator*()
        {
            return m_iter->payload;
        }

        iterator& operator++()
        {
            ++m_iter;
            return *this;
        }

        friend bool operator==(const iterator&, const iterator&) = default;

    private:
        typename SentinelLinkedList<Node>::iterator m_iter;
    };

    iterator begin() const { return iterator(const_cast<SentinelLinkedList<Node>&>(m_list).begin()); }
    iterator end() const { return iterator(const_cast<SentinelLinkedList<Node>&>(m_list).end()); }
    
private:
    Node* newNode(T value)
    {
        Node* result = m_nodes.add();
        result->payload = value;
        m_map.set(value, result);
        return result;
    }
    
    UncheckedKeyHashMap<T, Node*> m_map;
    Bag<Node> m_nodes; // FIXME: We could just manually free the contents of the linked list.
    SentinelLinkedList<Node> m_list;
};

} // namespace WTF

using WTF::OrderMaker;
