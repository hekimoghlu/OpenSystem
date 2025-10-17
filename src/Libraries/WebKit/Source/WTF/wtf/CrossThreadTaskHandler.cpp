/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#include <wtf/CrossThreadTaskHandler.h>

#include <wtf/AutodrainedPool.h>

namespace WTF {

CrossThreadTaskHandler::CrossThreadTaskHandler(ASCIILiteral threadName, AutodrainedPoolForRunLoop useAutodrainedPool)
    : m_useAutodrainedPool(useAutodrainedPool)
{
    ASSERT(isMainThread());
    Locker<Lock> locker(m_taskThreadCreationLock);
    Thread::create(threadName, [this] {
        taskRunLoop();

        if (m_completionCallback)
            m_completionCallback();
    })->detach();
}

CrossThreadTaskHandler::~CrossThreadTaskHandler()
{
    ASSERT(isMainThread());
}

void CrossThreadTaskHandler::postTask(CrossThreadTask&& task)
{
    m_taskQueue.append(WTFMove(task));
}

void CrossThreadTaskHandler::postTaskReply(CrossThreadTask&& task)
{
    m_taskReplyQueue.append(WTFMove(task));

    Locker locker { m_mainThreadReplyLock };
    if (m_mainThreadReplyScheduled)
        return;

    m_mainThreadReplyScheduled = true;
    callOnMainThread([this] {
        handleTaskRepliesOnMainThread();
    });
}

void CrossThreadTaskHandler::taskRunLoop()
{
    ASSERT(!isMainThread());
    {
        Locker<Lock> locker(m_taskThreadCreationLock);
    }

    while (auto task = m_taskQueue.waitForMessage()) {
        std::unique_ptr<AutodrainedPool> autodrainedPool = (m_useAutodrainedPool == AutodrainedPoolForRunLoop::Use) ? makeUnique<AutodrainedPool>() : nullptr;

        task.performTask();
    }
}

void CrossThreadTaskHandler::handleTaskRepliesOnMainThread()
{
    {
        Locker locker { m_mainThreadReplyLock };
        m_mainThreadReplyScheduled = false;
    }

    while (auto task = m_taskReplyQueue.tryGetMessage())
        task->performTask();
}

void CrossThreadTaskHandler::setCompletionCallback(Function<void ()>&& completionCallback)
{
    m_completionCallback = WTFMove(completionCallback);
}

void CrossThreadTaskHandler::kill()
{
    m_taskQueue.kill();
    m_taskReplyQueue.kill();
}

} // namespace WTF
