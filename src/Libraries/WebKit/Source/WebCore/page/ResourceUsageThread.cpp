/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
#include "ResourceUsageThread.h"

#if ENABLE(RESOURCE_USAGE)

#include "CommonVM.h"
#include <thread>
#include <wtf/MainThread.h>
#include <wtf/Vector.h>

namespace WebCore {

ResourceUsageThread::ResourceUsageThread()
{
}

ResourceUsageThread& ResourceUsageThread::singleton()
{
    static NeverDestroyed<ResourceUsageThread> resourceUsageThread;
    return resourceUsageThread;
}

void ResourceUsageThread::addObserver(void* key, ResourceUsageCollectionMode mode, std::function<void (const ResourceUsageData&)> function)
{
    auto& resourceUsageThread = ResourceUsageThread::singleton();
    resourceUsageThread.createThreadIfNeeded();

    {
        Locker locker { resourceUsageThread.m_observersLock };
        bool wasEmpty = resourceUsageThread.m_observers.isEmpty();
        resourceUsageThread.m_observers.set(key, std::make_pair(mode, function));

        resourceUsageThread.recomputeCollectionMode();

        if (wasEmpty) {
            resourceUsageThread.platformSaveStateBeforeStarting();
            resourceUsageThread.m_condition.notifyAll();
        }
    }
}

void ResourceUsageThread::removeObserver(void* key)
{
    auto& resourceUsageThread = ResourceUsageThread::singleton();

    {
        Locker locker { resourceUsageThread.m_observersLock };
        resourceUsageThread.m_observers.remove(key);

        resourceUsageThread.recomputeCollectionMode();
    }
}

void ResourceUsageThread::waitUntilObservers()
{
    Locker locker { m_observersLock };
    while (m_observers.isEmpty()) {
        m_condition.wait(m_observersLock);

        // Wait a bit after waking up for the first time.
        sleep(10_ms);
    }
}

void ResourceUsageThread::notifyObservers(ResourceUsageData&& data)
{
    callOnMainThread([data = WTFMove(data)]() mutable {
        Vector<std::pair<ResourceUsageCollectionMode, std::function<void (const ResourceUsageData&)>>> pairs;

        {
            auto& resourceUsageThread = ResourceUsageThread::singleton();
            Locker locker { resourceUsageThread.m_observersLock };
            pairs = copyToVector(resourceUsageThread.m_observers.values());
        }

        for (auto& pair : pairs)
            pair.second(data);
    });
}

void ResourceUsageThread::recomputeCollectionMode()
{
    m_collectionMode = None;

    for (auto& pair : m_observers.values())
        m_collectionMode = static_cast<ResourceUsageCollectionMode>(m_collectionMode | pair.first);
}

void ResourceUsageThread::createThreadIfNeeded()
{
    if (m_thread)
        return;

    m_vm = &commonVM();
    m_thread = Thread::create("WebCore: ResourceUsage"_s, [this] {
        threadBody();
    });
}

NO_RETURN void ResourceUsageThread::threadBody()
{
    // Wait a bit after waking up for the first time.
    sleep(10_ms);
    
    while (true) {
        // Only do work if we have observers.
        waitUntilObservers();

        auto start = WallTime::now();

        ResourceUsageData data;
        ResourceUsageCollectionMode mode = m_collectionMode;
        if (mode & CPU)
            platformCollectCPUData(m_vm, data);
        if (mode & Memory)
            platformCollectMemoryData(m_vm, data);

        notifyObservers(WTFMove(data));

        // NOTE: Web Inspector expects this interval to be 500ms (CPU / Memory timelines),
        // so if this interval changes Web Inspector may need to change.
        auto duration = WallTime::now() - start;
        auto difference = 500_ms - duration;
        sleep(difference);
    }
}

} // namespace WebCore

#endif // ENABLE(RESOURCE_USAGE)
