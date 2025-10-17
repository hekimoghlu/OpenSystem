/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#include "MicrotaskQueue.h"

#include "Debugger.h"
#include "JSGlobalObject.h"
#include "JSObject.h"
#include "SlotVisitorInlines.h"
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MicrotaskQueue);

bool QueuedTask::isRunnable() const
{
    if (RefPtr dispatcher = m_dispatcher)
        return dispatcher->isRunnable();
    return true;
}

MicrotaskQueue::MicrotaskQueue(VM& vm)
{
    vm.m_microtaskQueues.append(this);
}

MicrotaskQueue::~MicrotaskQueue()
{
    if (isOnList())
        remove();
}

template<typename Visitor>
void MicrotaskQueue::visitAggregateImpl(Visitor& visitor)
{
    m_queue.visitAggregate(visitor);
    m_toKeep.visitAggregate(visitor);
}
DEFINE_VISIT_AGGREGATE(MicrotaskQueue);

void MicrotaskQueue::enqueue(QueuedTask&& task)
{
    auto* globalObject = task.globalObject();
    auto microtaskIdentifier = task.identifier();
    m_queue.enqueue(WTFMove(task));
    if (globalObject) {
        if (auto* debugger = globalObject->debugger(); UNLIKELY(debugger))
            debugger->didQueueMicrotask(globalObject, microtaskIdentifier);
    }
}

bool MarkedMicrotaskDeque::hasMicrotasksForFullyActiveDocument() const
{
    for (auto& task : m_queue) {
        if (task.isRunnable())
            return true;
    }
    return false;
}

template<typename Visitor>
void MarkedMicrotaskDeque::visitAggregateImpl(Visitor& visitor)
{
    // Because content in the queue will not be changed, we need to scan it only once per an entry during one GC cycle.
    // We record the previous scan's index, and restart scanning again in CollectorPhase::FixPoint from that.
    // When new GC phase begins, this cursor is reset to zero (beginMarking). This optimization is introduced because
    // some of application have massive size of MicrotaskQueue depth. For example, in parallel-promises-es2015-native.js
    // benchmark, it becomes 251670 at most.
    // This cursor is adjusted when an entry is dequeued. And we do not use any locking here, and that's fine: these
    // values are read by GC when CollectorPhase::FixPoint and CollectorPhase::Begin, and both suspend the mutator, thus,
    // there is no concurrency issue.
    for (auto iterator = m_queue.begin() + m_markedBefore, end = m_queue.end(); iterator != end; ++iterator) {
        auto& task = *iterator;
        visitor.appendUnbarriered(task.m_globalObject);
        visitor.appendUnbarriered(task.m_job);
        visitor.appendUnbarriered(task.m_arguments, QueuedTask::maxArguments);
    }
    m_markedBefore = m_queue.size();
}
DEFINE_VISIT_AGGREGATE(MarkedMicrotaskDeque);

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
