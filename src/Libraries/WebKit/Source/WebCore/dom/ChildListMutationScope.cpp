/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#include "config.h"
#include "ChildListMutationScope.h"

#include "MutationObserverInterestGroup.h"
#include "MutationRecord.h"
#include "StaticNodeList.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

using AccumulatorMap = UncheckedKeyHashMap<WeakRef<ContainerNode, WeakPtrImplWithEventTargetData>, SingleThreadWeakRef<ChildListMutationAccumulator>>;
static AccumulatorMap& accumulatorMap()
{
    static NeverDestroyed<AccumulatorMap> map;
    return map;
}

ChildListMutationAccumulator::ChildListMutationAccumulator(ContainerNode& target, std::unique_ptr<MutationObserverInterestGroup> observers)
    : m_target(target)
    , m_observers(WTFMove(observers))
{
}

ChildListMutationAccumulator::~ChildListMutationAccumulator()
{
    if (!isEmpty())
        enqueueMutationRecord();
    accumulatorMap().remove(m_target.get());
}

Ref<ChildListMutationAccumulator> ChildListMutationAccumulator::getOrCreate(ContainerNode& target)
{
    RefPtr<ChildListMutationAccumulator> newAccumulator;
    auto addResult = accumulatorMap().ensure(target, [&]() -> SingleThreadWeakRef<ChildListMutationAccumulator> {
        newAccumulator = adoptRef(new ChildListMutationAccumulator(target, MutationObserverInterestGroup::createForChildListMutation(target)));
        return *newAccumulator;
    });
    if (newAccumulator)
        return newAccumulator.releaseNonNull();
    return addResult.iterator->value.get();
}

inline bool ChildListMutationAccumulator::isAddedNodeInOrder(Node& child)
{
    return isEmpty() || (m_lastAdded == child.previousSibling() && m_nextSibling == child.nextSibling());
}

void ChildListMutationAccumulator::childAdded(Node& child)
{
    ASSERT(hasObservers());

    Ref protectedChild { child }; // FIXME: The call sites should be refing the child, not us.

    if (!isAddedNodeInOrder(child))
        enqueueMutationRecord();

    if (isEmpty()) {
        m_previousSibling = child.previousSibling();
        m_nextSibling = child.nextSibling();
    }

    m_lastAdded = &child;
    m_addedNodes.append(child);
}

inline bool ChildListMutationAccumulator::isRemovedNodeInOrder(Node& child)
{
    return isEmpty() || m_nextSibling == &child;
}

void ChildListMutationAccumulator::willRemoveChild(Node& child)
{
    ASSERT(hasObservers());

    Ref protectedChild { child }; // FIXME: The call sites should be refing the child, not us.

    if (!m_addedNodes.isEmpty() || !isRemovedNodeInOrder(child))
        enqueueMutationRecord();

    if (isEmpty()) {
        m_previousSibling = child.previousSibling();
        m_nextSibling = child.nextSibling();
        m_lastAdded = child.previousSibling();
    } else
        m_nextSibling = child.nextSibling();

    m_removedNodes.append(child);
}

void ChildListMutationAccumulator::enqueueMutationRecord()
{
    ASSERT(hasObservers());
    ASSERT(!isEmpty());

    Ref record = MutationRecord::createChildList(protectedTarget(), StaticNodeList::create(WTFMove(m_addedNodes)), StaticNodeList::create(WTFMove(m_removedNodes)), WTFMove(m_previousSibling), WTFMove(m_nextSibling));
    m_observers->enqueueMutationRecord(WTFMove(record));
    m_lastAdded = nullptr;
    ASSERT(isEmpty());
}

Ref<ContainerNode> ChildListMutationAccumulator::protectedTarget() const
{
    return m_target;
}

bool ChildListMutationAccumulator::isEmpty()
{
    bool result = m_removedNodes.isEmpty() && m_addedNodes.isEmpty();
#ifndef NDEBUG
    if (result) {
        ASSERT(!m_previousSibling);
        ASSERT(!m_nextSibling);
        ASSERT(!m_lastAdded);
    }
#endif
    return result;
}

} // namespace WebCore
