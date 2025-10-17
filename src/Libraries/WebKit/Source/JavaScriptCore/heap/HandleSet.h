/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#include "Handle.h"
#include "HandleBlock.h"
#include "HeapCell.h"
#include <wtf/DoublyLinkedList.h>
#include <wtf/HashCountedSet.h>
#include <wtf/SentinelLinkedList.h>
#include <wtf/SinglyLinkedList.h>

namespace JSC {

class HandleSet;
class VM;
class JSValue;

class HandleNode final : public BasicRawSentinelNode<HandleNode> {
public:
    HandleNode() = default;
    
    HandleSlot slot();
    HandleSet* handleSet();

    static HandleNode* toHandleNode(HandleSlot slot)
    {
        return std::bit_cast<HandleNode*>(std::bit_cast<uintptr_t>(slot) - OBJECT_OFFSETOF(HandleNode, m_value));
    }

private:
    JSValue m_value { };
};

class HandleSet {
    friend class HandleBlock;
public:
    static HandleSet* heapFor(HandleSlot);

    HandleSet(VM&);
    ~HandleSet();

    VM& vm();

    HandleSlot allocate();
    void deallocate(HandleSlot);

    template<typename Visitor> void visitStrongHandles(Visitor&);

    template<bool isCellOnly>
    void writeBarrier(HandleSlot, JSValue);

    unsigned protectedGlobalObjectCount();

    template<typename Functor> void forEachStrongHandle(const Functor&, const HashCountedSet<JSCell*>& skipSet);

private:
    typedef HandleNode Node;

    JS_EXPORT_PRIVATE void grow();
    
#if ENABLE(GC_VALIDATION) || ASSERT_ENABLED
    JS_EXPORT_PRIVATE bool isLiveNode(Node*);
#endif

    VM& m_vm;
    DoublyLinkedList<HandleBlock> m_blockList;

    using NodeList = SentinelLinkedList<Node, BasicRawSentinelNode<Node>>;
    NodeList m_strongList;
    SinglyLinkedList<Node> m_freeList;
};

inline HandleSet* HandleSet::heapFor(HandleSlot handle)
{
    return HandleNode::toHandleNode(handle)->handleSet();
}

inline VM& HandleSet::vm()
{
    return m_vm;
}

inline HandleSlot HandleSet::allocate()
{
    if (m_freeList.isEmpty())
        grow();

    HandleSet::Node* node = m_freeList.pop();
    new (NotNull, node) HandleSet::Node();
    return node->slot();
}

inline void HandleSet::deallocate(HandleSlot handle)
{
    HandleSet::Node* node = HandleNode::toHandleNode(handle);
    if (node->isOnList())
        NodeList::remove(node);
    m_freeList.push(node);
}

inline HandleSlot HandleNode::slot()
{
    return &m_value;
}

inline HandleSet* HandleNode::handleSet()
{
    return HandleBlock::blockFor(this)->handleSet();
}

template<typename Functor> void HandleSet::forEachStrongHandle(const Functor& functor, const HashCountedSet<JSCell*>& skipSet)
{
    for (Node& node : m_strongList) {
        JSValue value = *node.slot();
        if (!value || !value.isCell())
            continue;
        if (skipSet.contains(value.asCell()))
            continue;
        functor(value.asCell());
    }
}

template<bool isCellOnly>
inline void HandleSet::writeBarrier(HandleSlot slot, JSValue value)
{
    bool valueIsNonEmptyCell = value && (isCellOnly || value.isCell());
    bool slotIsNonEmptyCell = *slot && (isCellOnly || slot->isCell());
    if (valueIsNonEmptyCell == slotIsNonEmptyCell)
        return;

    Node* node = HandleNode::toHandleNode(slot);
#if ENABLE(GC_VALIDATION)
    if (node->isOnList())
        RELEASE_ASSERT(isLiveNode(node));
#endif
    if (!valueIsNonEmptyCell) {
        ASSERT(slotIsNonEmptyCell);
        ASSERT(node->isOnList());
        NodeList::remove(node);
        return;
    }

    ASSERT(!slotIsNonEmptyCell);
    ASSERT(!node->isOnList());
    m_strongList.push(node);

#if ENABLE(GC_VALIDATION)
    RELEASE_ASSERT(isLiveNode(node));
#endif
}

} // namespace JSC
