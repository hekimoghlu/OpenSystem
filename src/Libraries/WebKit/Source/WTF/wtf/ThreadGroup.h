/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

#include <memory>
#include <wtf/ListHashSet.h>
#include <wtf/Lock.h>
#include <wtf/Threading.h>

namespace WTF {

enum class ThreadGroupAddResult { NewlyAdded, AlreadyAdded, NotAdded };

class ThreadGroup final : public std::enable_shared_from_this<ThreadGroup> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(ThreadGroup);
public:
    friend class Thread;

    static std::shared_ptr<ThreadGroup> create()
    {
        return std::allocate_shared<ThreadGroup>(FastAllocator<ThreadGroup>());
    }

    WTF_EXPORT_PRIVATE ThreadGroupAddResult add(Thread&);
    WTF_EXPORT_PRIVATE ThreadGroupAddResult add(const AbstractLocker&, Thread&);
    WTF_EXPORT_PRIVATE ThreadGroupAddResult addCurrentThread();

    const ListHashSet<Ref<Thread>>& threads(const AbstractLocker&) const { return m_threads; }

    WordLock& getLock() { return m_lock; }

    WTF_EXPORT_PRIVATE ~ThreadGroup();

    ThreadGroup() = default;

private:
    std::weak_ptr<ThreadGroup> weakFromThis()
    {
        return shared_from_this();
    }

    // We use WordLock since it can be used when deallocating TLS.
    WordLock m_lock;
    ListHashSet<Ref<Thread>> m_threads;
};

}

using WTF::ThreadGroup;
using WTF::ThreadGroupAddResult;
