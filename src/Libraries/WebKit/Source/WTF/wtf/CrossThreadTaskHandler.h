/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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

#include <wtf/CrossThreadQueue.h>
#include <wtf/CrossThreadTask.h>
#include <wtf/Lock.h>
#include <wtf/Threading.h>

namespace WTF {

class RegistrationStore;
class SQLiteDatabase;

class CrossThreadTaskHandler {
public:
    WTF_EXPORT_PRIVATE virtual ~CrossThreadTaskHandler();
    enum class AutodrainedPoolForRunLoop { DoNotUse, Use };

protected:
    WTF_EXPORT_PRIVATE CrossThreadTaskHandler(ASCIILiteral threadName, AutodrainedPoolForRunLoop = AutodrainedPoolForRunLoop::DoNotUse);

    WTF_EXPORT_PRIVATE void postTask(CrossThreadTask&&);
    WTF_EXPORT_PRIVATE void postTaskReply(CrossThreadTask&&);

    WTF_EXPORT_PRIVATE void kill();
    WTF_EXPORT_PRIVATE void setCompletionCallback(Function<void ()>&&);

private:
    void handleTaskRepliesOnMainThread();
    void taskRunLoop();

    AutodrainedPoolForRunLoop m_useAutodrainedPool { AutodrainedPoolForRunLoop::DoNotUse };

    Lock m_taskThreadCreationLock;
    Lock m_mainThreadReplyLock;
    bool m_mainThreadReplyScheduled WTF_GUARDED_BY_LOCK(m_mainThreadReplyLock) { false };

    CrossThreadQueue<CrossThreadTask> m_taskQueue;
    CrossThreadQueue<CrossThreadTask> m_taskReplyQueue;

    Function<void ()> m_completionCallback;
};

} // namespace WTF

using WTF::CrossThreadTaskHandler;

