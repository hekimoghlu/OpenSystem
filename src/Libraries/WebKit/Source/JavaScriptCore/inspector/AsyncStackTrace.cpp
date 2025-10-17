/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include "AsyncStackTrace.h"

#include "ScriptCallStack.h"
#include <wtf/Ref.h>

namespace Inspector {

Ref<AsyncStackTrace> AsyncStackTrace::create(Ref<ScriptCallStack>&& callStack, bool singleShot, RefPtr<AsyncStackTrace> parent)
{
    return adoptRef(*new AsyncStackTrace(WTFMove(callStack), singleShot, WTFMove(parent)));
}

AsyncStackTrace::AsyncStackTrace(Ref<ScriptCallStack>&& callStack, bool singleShot, RefPtr<AsyncStackTrace> parent)
    : m_callStack(WTFMove(callStack))
    , m_parent(parent)
    , m_singleShot(singleShot)
{
    ASSERT(size());

    if (m_parent)
        m_parent->m_childCount++;
}

AsyncStackTrace::~AsyncStackTrace()
{
    if (m_parent)
        remove();
    ASSERT(!m_childCount);
}

bool AsyncStackTrace::isPending() const
{
    return m_state == State::Pending;
}

bool AsyncStackTrace::isLocked() const
{
    return m_state == State::Pending || m_state == State::Active || m_childCount > 1;
}

const ScriptCallFrame& AsyncStackTrace::at(size_t index) const
{
    return m_callStack->at(index);
}

size_t AsyncStackTrace::size() const
{
    return m_callStack->size();
}

bool AsyncStackTrace::topCallFrameIsBoundary() const
{
    return at(0).isNative();
}

void AsyncStackTrace::willDispatchAsyncCall(size_t maxDepth)
{
    ASSERT(m_state == State::Pending);
    m_state = State::Active;

    truncate(maxDepth);
}

void AsyncStackTrace::didDispatchAsyncCall()
{
    ASSERT(m_state == State::Active || m_state == State::Canceled);

    if (m_state == State::Active && !m_singleShot) {
        m_state = State::Pending;
        return;
    }

    m_state = State::Dispatched;

    if (!m_childCount)
        remove();
}

void AsyncStackTrace::didCancelAsyncCall()
{
    if (m_state == State::Canceled)
        return;

    if (m_state == State::Pending && !m_childCount)
        remove();

    m_state = State::Canceled;
}

Ref<Protocol::Console::StackTrace> AsyncStackTrace::buildInspectorObject() const
{
    RefPtr<Protocol::Console::StackTrace> topStackTrace;
    RefPtr<Protocol::Console::StackTrace> previousStackTrace;

    auto* stackTrace = this;
    while (stackTrace) {
        auto& callStack = stackTrace->m_callStack;

        auto protocolObject = Protocol::Console::StackTrace::create()
            .setCallFrames(callStack->buildInspectorArray())
            .release();

        if (stackTrace->m_truncated)
            protocolObject->setTruncated(true);
        if (stackTrace->topCallFrameIsBoundary())
            protocolObject->setTopCallFrameIsBoundary(true);

        if (!topStackTrace)
            topStackTrace = protocolObject.ptr();

        if (previousStackTrace)
            previousStackTrace->setParentStackTrace(protocolObject.copyRef());

        previousStackTrace = WTFMove(protocolObject);
        stackTrace = stackTrace->m_parent.get();
    }

    return topStackTrace.releaseNonNull();
}

void AsyncStackTrace::truncate(size_t maxDepth)
{
    AsyncStackTrace* lastUnlockedAncestor = nullptr;
    size_t depth = 0;

    auto* newStackTraceRoot = this;
    while (newStackTraceRoot) {
        depth += newStackTraceRoot->size();
        if (depth >= maxDepth)
            break;

        auto* parent = newStackTraceRoot->m_parent.get();
        if (!lastUnlockedAncestor && parent && parent->isLocked())
            lastUnlockedAncestor = newStackTraceRoot;

        newStackTraceRoot = parent;
    }

    if (!newStackTraceRoot || !newStackTraceRoot->m_parent)
        return;

    if (!lastUnlockedAncestor) {
        // No locked nodes belong to the trace. The subtree at the new root
        // is moved to a new tree, and marked as truncated if necessary.
        newStackTraceRoot->m_truncated = true;
        newStackTraceRoot->remove();
        return;
    }

    // The new root has a locked descendent. Since truncating a stack trace
    // cannot mutate locked nodes or their ancestors, a new tree is created by
    // cloning the locked portion of the trace (the path from the locked node
    // to the new root). The subtree rooted at the last unlocked ancestor is
    // then appended to the new tree.
    auto* previousNode = lastUnlockedAncestor;

    // The subtree being truncated must be removed from it's parent before
    // updating its parent pointer chain.
    RefPtr<AsyncStackTrace> sourceNode = lastUnlockedAncestor->m_parent;
    lastUnlockedAncestor->remove();

    while (sourceNode) {
        previousNode->m_parent = AsyncStackTrace::create(sourceNode->m_callStack.copyRef(), true, nullptr);
        previousNode->m_parent->m_childCount = 1;
        previousNode = previousNode->m_parent.get();

        if (sourceNode.get() == newStackTraceRoot)
            break;

        sourceNode = sourceNode->m_parent;
    }

    previousNode->m_truncated = true;
}

void AsyncStackTrace::remove()
{
    if (!m_parent)
        return;

    ASSERT(m_parent->m_childCount);
    m_parent->m_childCount--;
    m_parent = nullptr;

    m_callStack->removeParentStackTrace();
}

} // namespace Inspector
