/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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

#if ENABLE(RESOURCE_USAGE)

#include "ResourceUsageData.h"
#include <array>
#include <functional>
#include <wtf/Condition.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Noncopyable.h>
#include <wtf/Threading.h>

#if OS(DARWIN)
#include <mach/mach.h>
#endif

namespace JSC {
class VM;
}

namespace WebCore {

enum ResourceUsageCollectionMode {
    None = 0,
    CPU = 1 << 0,
    Memory = 1 << 1,
    All = CPU | Memory,
};

class ResourceUsageThread {
    WTF_MAKE_NONCOPYABLE(ResourceUsageThread);

public:
    static void addObserver(void* key, ResourceUsageCollectionMode, std::function<void (const ResourceUsageData&)>);
    static void removeObserver(void* key);

private:
    friend NeverDestroyed<ResourceUsageThread>;
    ResourceUsageThread();
    static ResourceUsageThread& singleton();

    void waitUntilObservers();
    void notifyObservers(ResourceUsageData&&);

    void recomputeCollectionMode() WTF_REQUIRES_LOCK(m_observersLock);

    void createThreadIfNeeded();
    NO_RETURN void threadBody();

    void platformSaveStateBeforeStarting();
    void platformCollectCPUData(JSC::VM*, ResourceUsageData&);
    void platformCollectMemoryData(JSC::VM*, ResourceUsageData&);

    RefPtr<Thread> m_thread;
    Lock m_observersLock;
    Condition m_condition;
    HashMap<void*, std::pair<ResourceUsageCollectionMode, std::function<void(const ResourceUsageData&)>>> m_observers WTF_GUARDED_BY_LOCK(m_observersLock);
    ResourceUsageCollectionMode m_collectionMode { None };

    // Platforms may need to access some data from the common VM.
    // They should ensure their use of the VM is thread safe.
    JSC::VM* m_vm { nullptr };

#if ENABLE(SAMPLING_PROFILER)
#if OS(DARWIN)
    mach_port_t m_samplingProfilerMachThread { MACH_PORT_NULL };
#elif OS(LINUX)
    pid_t m_samplingProfilerThreadID { 0 };
#endif
#endif

};

} // namespace WebCore

#endif // ENABLE(RESOURCE_USAGE)
