/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "WebCoreThreadRun.h"

#if PLATFORM(IOS_FAMILY)

#include "WebCoreThreadInternal.h"
#include <mutex>
#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/Vector.h>

namespace {

class WebThreadBlockState {
public:
    WebThreadBlockState()
        : m_completed(false)
    {
    }

    void waitForCompletion()
    {
        Locker lock { m_stateMutex };

        m_completionConditionVariable.wait(m_stateMutex, [this] { return m_completed; });
    }

    void setCompleted()
    {
        Locker locker { m_stateMutex };

        ASSERT(!m_completed);
        m_completed = true;
        m_completionConditionVariable.notifyOne();
    }

private:
    Lock m_stateMutex;
    Condition m_completionConditionVariable;
    bool m_completed;
};

class WebThreadBlock {
public:
    WebThreadBlock(void (^block)(void), WebThreadBlockState* state)
        : m_block(Block_copy(block))
        , m_state(state)
    {
    }

    WebThreadBlock(const WebThreadBlock& other)
        : m_block(Block_copy(other.m_block))
        , m_state(other.m_state)
    {
    }

    WebThreadBlock& operator=(const WebThreadBlock& other)
    {
        void (^oldBlock)() = m_block;
        m_block = Block_copy(other.m_block);
        Block_release(oldBlock);
        m_state = other.m_state;
        return *this;
    }

    ~WebThreadBlock()
    {
        Block_release(m_block);
    }

    void operator()() const
    {
        m_block();
        if (m_state)
            m_state->setCompleted();
    }

private:
    void (^m_block)(void);
    WebThreadBlockState* m_state;
};

}

extern "C" {

typedef Vector<WebThreadBlock> WebThreadRunQueue;

static Lock runQueueMutex;
static WebThreadRunQueue* runQueue;

static RetainPtr<CFRunLoopSourceRef>& runSource()
{
    static NeverDestroyed<RetainPtr<CFRunLoopSourceRef>> runSource;
    return runSource;
}

static void HandleRunSource(void *info)
{
    UNUSED_PARAM(info);
    ASSERT(WebThreadIsCurrent());
    ASSERT(runSource());
    ASSERT(runQueue);

    WebThreadRunQueue queueCopy;
    {
        Locker locker { runQueueMutex };
        queueCopy = *runQueue;
        runQueue->clear();
    }

    for (const auto& block : queueCopy)
        block();
}

static void _WebThreadRun(void (^block)(void), bool synchronous)
{
    if (WebThreadIsCurrent() || !WebThreadIsEnabled()) {
        block();
        return;
    }

    ASSERT(runSource());
    ASSERT(runQueue);

    WebThreadBlockState* state = 0;
    if (synchronous)
        state = new WebThreadBlockState;

    {
        Locker locker { runQueueMutex };
        runQueue->append(WebThreadBlock(block, state));
    }

    CFRunLoopSourceSignal(runSource().get());
    CFRunLoopWakeUp(WebThreadRunLoop());

    if (synchronous) {
        state->waitForCompletion();
        delete state;
    }
}

void WebThreadRun(void (^block)(void))
{
    _WebThreadRun(block, false);
}

void WebThreadInitRunQueue()
{
    ASSERT(!runQueue);
    ASSERT(!runSource());

    static dispatch_once_t pred;
    dispatch_once(&pred, ^{
        runQueue = new WebThreadRunQueue;

        CFRunLoopSourceContext runSourceContext = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, HandleRunSource };
        runSource() = adoptCF(CFRunLoopSourceCreate(nullptr, -1, &runSourceContext));
        CFRunLoopAddSource(WebThreadRunLoop(), runSource().get(), kCFRunLoopDefaultMode);
    });
}

}

#endif // PLATFORM(IOS_FAMILY)
