/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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

#include <wtf/FastMalloc.h>
#include <wtf/Noncopyable.h>
#include <wtf/Packed.h>
#include <wtf/RawPtrTraits.h>

namespace WTF {

template<typename T, typename PassedPtrTraits = RawPtrTraits<T>>
class BagNode {
public:
    using PtrTraits = typename PassedPtrTraits::template RebindTraits<BagNode>;

    template<typename... Args>
    BagNode(Args&&... args)
        : m_item(std::forward<Args>(args)...)
    { }
    
    T m_item;
    typename PtrTraits::StorageType m_next { nullptr };
};

template<typename T, typename PassedPtrTraits = RawPtrTraits<T>, typename Malloc = FastMalloc>
class Bag final {
    WTF_MAKE_NONCOPYABLE(Bag);
    WTF_MAKE_FAST_ALLOCATED;
    using Node = BagNode<T, PassedPtrTraits>;
    using PtrTraits = typename PassedPtrTraits::template RebindTraits<Node>;

public:
    Bag() = default;

    template<typename U>
    Bag(Bag<T, U>&& other)
    {
        ASSERT(!m_head);
        m_head = other.unwrappedHead();
        other.m_head = nullptr;
    }

    template<typename U>
    Bag& operator=(Bag<T, U>&& other)
    {
        if (unwrappedHead() == other.unwrappedHead())
            return *this;

        Bag destroy;
        destroy.m_head = unwrappedHead();
        m_head = other.unwrappedHead();
        other.m_head = nullptr;

        return *this;
    }

    ~Bag()
    {
        clear();
    }
    
    void clear()
    {
        Node* head = this->unwrappedHead();
        while (head) {
            Node* current = head;
            head = Node::PtrTraits::unwrap(current->m_next);
            current->~Node();
            Malloc::free(current);
        }
        m_head = nullptr;
    }
    
    template<typename... Args>
    T* add(Args&&... args)
    {
        Node* newNode = static_cast<Node*>(Malloc::malloc(sizeof(Node)));
        new (NotNull, newNode) Node(std::forward<Args>(args)...);
        newNode->m_next = unwrappedHead();
        m_head = newNode;
        return &newNode->m_item;
    }
    
    class iterator {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        iterator()
            : m_node(0)
        {
        }
        
        // This is sort of cheating; it's equivalent to iter == end().
        bool operator!() const { return !m_node; }
        
        T* operator*() const { return &m_node->m_item; }
        
        iterator& operator++()
        {
            m_node = Node::PtrTraits::unwrap(m_node->m_next);
            return *this;
        }
        
        friend bool operator==(iterator, iterator) = default;

    private:
        template<typename, typename, typename> friend class WTF::Bag;
        Node* m_node;
    };
    
    iterator begin()
    {
        iterator result;
        result.m_node = unwrappedHead();
        return result;
    }

    const iterator begin() const
    {
        iterator result;
        result.m_node = unwrappedHead();
        return result;
    }


    iterator end() const { return iterator(); }
    
    bool isEmpty() const { return !m_head; }
    
private:
    Node* unwrappedHead() const { return PtrTraits::unwrap(m_head); }

    typename PtrTraits::StorageType m_head { nullptr };
};

template<typename T>
using PackedBag = Bag<T, PackedPtrTraits<T>>;

} // namespace WTF

using WTF::Bag;
using WTF::PackedBag;
