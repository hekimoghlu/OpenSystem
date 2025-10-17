/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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

#include <wtf/Function.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WTF {

class CancellableTask;

class TaskCancellationGroupImpl final : public RefCountedAndCanMakeWeakPtr<TaskCancellationGroupImpl> {
public:
    static Ref<TaskCancellationGroupImpl> create()
    {
        return adoptRef(*new TaskCancellationGroupImpl);
    }
    void cancel() { weakPtrFactory().revokeAll(); }
    bool hasPendingTask() const { return weakPtrFactory().weakPtrCount(); }

private:
    TaskCancellationGroupImpl() = default;
};

class TaskCancellationGroupHandle {
public:
    bool isCancelled() const { return !m_impl; }
    void clear() { m_impl = nullptr; }
private:
    friend class TaskCancellationGroup;
    explicit TaskCancellationGroupHandle(TaskCancellationGroupImpl& impl)
        : m_impl(impl)
    {
    }
    WeakPtr<TaskCancellationGroupImpl> m_impl;
};

class TaskCancellationGroup {
public:
    TaskCancellationGroup()
        : m_impl(TaskCancellationGroupImpl::create())
    {
    }
    void cancel() { m_impl->cancel(); }
    bool hasPendingTask() const { return m_impl->hasPendingTask(); }

private:
    friend class CancellableTask;
    TaskCancellationGroupHandle createHandle() { return TaskCancellationGroupHandle { m_impl }; }

    Ref<TaskCancellationGroupImpl> m_impl;
};

class CancellableTask {
public:
    CancellableTask(TaskCancellationGroup&, Function<void()>&&);
    void operator()();

private:
    TaskCancellationGroupHandle m_cancellationGroup;
    Function<void()> m_task;
};

inline CancellableTask::CancellableTask(TaskCancellationGroup& cancellationGroup, Function<void()>&& task)
    : m_cancellationGroup(cancellationGroup.createHandle())
    , m_task(WTFMove(task))
{ }

inline void CancellableTask::operator()()
{
    if (m_cancellationGroup.isCancelled())
        return;

    m_cancellationGroup.clear();
    m_task();
}

} // namespace WTF

using WTF::CancellableTask;
using WTF::TaskCancellationGroup;
