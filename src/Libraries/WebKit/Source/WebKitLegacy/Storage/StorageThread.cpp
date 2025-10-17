/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#include "StorageThread.h"

#include <wtf/AutodrainedPool.h>
#include <wtf/HashSet.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static HashSet<CheckedRef<StorageThread>>& activeStorageThreads()
{
    ASSERT(isMainThread());
    static NeverDestroyed<HashSet<CheckedRef<StorageThread>>> threads;
    return threads;
}

StorageThread::StorageThread(Type type)
    : m_type(type)
{
    ASSERT(isMainThread());
}

StorageThread::~StorageThread()
{
    ASSERT(isMainThread());
    ASSERT(!m_thread);
}

void StorageThread::start()
{
    ASSERT(isMainThread());
    if (!m_thread) {
        if (m_type == Type::LocalStorage) {
            m_thread = Thread::create("LocalStorage"_s, [this] {
                threadEntryPoint();
            });
        } else {
            ASSERT(m_type == Type::IndexedDB);
            m_thread = Thread::create("IndexedDB"_s, [this] {
                threadEntryPoint();
            });
        }
    }
    activeStorageThreads().add(*this);
}

void StorageThread::threadEntryPoint()
{
    ASSERT(!isMainThread());

    while (auto function = m_queue.waitForMessage()) {
        AutodrainedPool pool;
        (*function)();
    }
}

void StorageThread::dispatch(Function<void ()>&& function)
{
    ASSERT(isMainThread());
    ASSERT(!m_queue.killed() && m_thread);
    m_queue.append(makeUnique<Function<void ()>>(WTFMove(function)));
}

void StorageThread::terminate()
{
    ASSERT(isMainThread());
    ASSERT(!m_queue.killed() && m_thread);
    activeStorageThreads().remove(*this);
    // Even in weird, exceptional cases, don't wait on a nonexistent thread to terminate.
    if (!m_thread)
        return;

    m_queue.append(makeUnique<Function<void ()>>([this] {
        performTerminate();
    }));
    m_thread->waitForCompletion();
    ASSERT(m_queue.killed());
    m_thread = nullptr;
}

void StorageThread::performTerminate()
{
    ASSERT(!isMainThread());
    m_queue.kill();
}

void StorageThread::releaseFastMallocFreeMemoryInAllThreads()
{
    for (auto& thread : activeStorageThreads())
        thread->dispatch(&WTF::releaseFastMallocFreeMemory);
}

}
