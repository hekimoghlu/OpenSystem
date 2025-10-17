/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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

#include "ContainerNode.h"
#include "DocumentInlines.h"
#include "MutationObserver.h"
#include <memory>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class MutationObserverInterestGroup;

// ChildListMutationAccumulator is not meant to be used directly; ChildListMutationScope is the public interface.
class ChildListMutationAccumulator : public RefCounted<ChildListMutationAccumulator>, public CanMakeSingleThreadWeakPtr<ChildListMutationAccumulator> {
public:
    static Ref<ChildListMutationAccumulator> getOrCreate(ContainerNode&);
    ~ChildListMutationAccumulator();

    void childAdded(Node&);
    void willRemoveChild(Node&);

    bool hasObservers() const { return !!m_observers; }

private:
    ChildListMutationAccumulator(ContainerNode&, std::unique_ptr<MutationObserverInterestGroup>);

    void enqueueMutationRecord();
    bool isEmpty();
    bool isAddedNodeInOrder(Node&);
    bool isRemovedNodeInOrder(Node&);

    Ref<ContainerNode> protectedTarget() const;

    Ref<ContainerNode> m_target;

    Vector<Ref<Node>> m_removedNodes;
    Vector<Ref<Node>> m_addedNodes;
    RefPtr<Node> m_previousSibling;
    RefPtr<Node> m_nextSibling;
    WeakPtr<Node, WeakPtrImplWithEventTargetData> m_lastAdded;

    std::unique_ptr<MutationObserverInterestGroup> m_observers;
};

class ChildListMutationScope {
    WTF_MAKE_NONCOPYABLE(ChildListMutationScope);
public:
    explicit ChildListMutationScope(ContainerNode& target)
    {
        if (target.document().hasMutationObserversOfType(MutationObserverOptionType::ChildList))
            m_accumulator = ChildListMutationAccumulator::getOrCreate(target);
    }

    bool canObserve() const { return m_accumulator; }

    void childAdded(Node& child)
    {
        if (m_accumulator && m_accumulator->hasObservers())
            m_accumulator->childAdded(child);
    }

    void willRemoveChild(Node& child)
    {
        if (m_accumulator && m_accumulator->hasObservers())
            m_accumulator->willRemoveChild(child);
    }

private:
    RefPtr<ChildListMutationAccumulator> m_accumulator;
};

} // namespace WebCore
