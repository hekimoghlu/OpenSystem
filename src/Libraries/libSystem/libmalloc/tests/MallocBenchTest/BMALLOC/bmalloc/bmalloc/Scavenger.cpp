/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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
#include "Scavenger.h"

#include "AllIsoHeapsInlines.h"
#include "AvailableMemory.h"
#include "BulkDecommit.h"
#include "Environment.h"
#include "Heap.h"
#if BOS(DARWIN)
#import <dispatch/dispatch.h>
#import <mach/host_info.h>
#import <mach/mach.h>
#import <mach/mach_error.h>
#endif
#include <stdio.h>
#include <thread>

namespace bmalloc {

static constexpr bool verbose = false;

struct PrintTime {
    PrintTime(const char* str) 
        : string(str)
    { }

    ~PrintTime()
    {
        if (!printed)
            print();
    }
    void print()
    {
        if (verbose) {
            fprintf(stderr, "%s %lfms\n", string, static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()) / 1000);
            printed = true;
        }
    }
    const char* string;
    std::chrono::steady_clock::time_point start { std::chrono::steady_clock::now() };
    bool printed { false };
};

Scavenger::Scavenger(std::lock_guard<Mutex>&)
{
#if BOS(DARWIN)
    auto queue = dispatch_queue_create("WebKit Malloc Memory Pressure Handler", DISPATCH_QUEUE_SERIAL);
    m_pressureHandlerDispatchSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0, DISPATCH_MEMORYPRESSURE_CRITICAL, queue);
    dispatch_source_set_event_handler(m_pressureHandlerDispatchSource, ^{
        scavenge();
    });
    dispatch_resume(m_pressureHandlerDispatchSource);
    dispatch_release(queue);
#endif
    
    m_thread = std::thread(&threadEntryPoint, this);
}

void Scavenger::run()
{
    std::lock_guard<Mutex> lock(m_mutex);
    runHoldingLock();
}

void Scavenger::runHoldingLock()
{
    m_state = State::Run;
    m_condition.notify_all();
}

void Scavenger::runSoon()
{
    std::lock_guard<Mutex> lock(m_mutex);
    runSoonHoldingLock();
}

void Scavenger::runSoonHoldingLock()
{
    if (willRunSoon())
        return;
    m_state = State::RunSoon;
    m_condition.notify_all();
}

void Scavenger::didStartGrowing()
{
    // We don't really need to lock here, since this is just a heuristic.
    m_isProbablyGrowing = true;
}

void Scavenger::scheduleIfUnderMemoryPressure(size_t bytes)
{
    std::lock_guard<Mutex> lock(m_mutex);
    scheduleIfUnderMemoryPressureHoldingLock(bytes);
}

void Scavenger::scheduleIfUnderMemoryPressureHoldingLock(size_t bytes)
{
    m_scavengerBytes += bytes;
    if (m_scavengerBytes < scavengerBytesPerMemoryPressureCheck)
        return;

    m_scavengerBytes = 0;

    if (willRun())
        return;

    if (!isUnderMemoryPressure())
        return;

    m_isProbablyGrowing = false;
    runHoldingLock();
}

void Scavenger::schedule(size_t bytes)
{
    std::lock_guard<Mutex> lock(m_mutex);
    scheduleIfUnderMemoryPressureHoldingLock(bytes);
    
    if (willRunSoon())
        return;
    
    m_isProbablyGrowing = false;
    runSoonHoldingLock();
}

inline void dumpStats()
{
    auto dump = [] (auto* string, auto size) {
        fprintf(stderr, "%s %zuMB\n", string, static_cast<size_t>(size) / 1024 / 1024);
    };

#if BOS(DARWIN)
    task_vm_info_data_t vmInfo;
    mach_msg_type_number_t vmSize = TASK_VM_INFO_COUNT;
    if (KERN_SUCCESS == task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)(&vmInfo), &vmSize)) {
        dump("phys_footprint", vmInfo.phys_footprint);
        dump("internal+compressed", vmInfo.internal + vmInfo.compressed);
    }
#endif

    dump("bmalloc-freeable", PerProcess<Scavenger>::get()->freeableMemory());
    dump("bmalloc-footprint", PerProcess<Scavenger>::get()->footprint());
}

std::chrono::milliseconds Scavenger::timeSinceLastFullScavenge()
{
    std::unique_lock<Mutex> lock(m_mutex);
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_lastFullScavengeTime);
}

std::chrono::milliseconds Scavenger::timeSinceLastPartialScavenge()
{
    std::unique_lock<Mutex> lock(m_mutex);
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_lastPartialScavengeTime);
}

void Scavenger::enableMiniMode()
{
    m_isInMiniMode = true; // We just store to this racily. The scavenger thread will eventually pick up the right value.
    if (m_state == State::RunSoon)
        run();
}

void Scavenger::scavenge()
{
    std::unique_lock<Mutex> lock(m_scavengingMutex);

    if (verbose) {
        fprintf(stderr, "--------------------------------\n");
        fprintf(stderr, "--before scavenging--\n");
        dumpStats();
    }

    {
        BulkDecommit decommitter;

        {
            PrintTime printTime("\nfull scavenge under lock time");
            std::lock_guard<Mutex> lock(Heap::mutex());
            for (unsigned i = numHeaps; i--;) {
                if (!isActiveHeapKind(static_cast<HeapKind>(i)))
                    continue;
                PerProcess<PerHeapKind<Heap>>::get()->at(i).scavenge(lock, decommitter);
            }
            decommitter.processEager();
        }

        {
            PrintTime printTime("full scavenge lazy decommit time");
            decommitter.processLazy();
        }

        {
            PrintTime printTime("full scavenge mark all as eligible time");
            std::lock_guard<Mutex> lock(Heap::mutex());
            for (unsigned i = numHeaps; i--;) {
                if (!isActiveHeapKind(static_cast<HeapKind>(i)))
                    continue;
                PerProcess<PerHeapKind<Heap>>::get()->at(i).markAllLargeAsEligibile(lock);
            }
        }
    }

    {
        RELEASE_BASSERT(!m_deferredDecommits.size());
        PerProcess<AllIsoHeaps>::get()->forEach(
            [&] (IsoHeapImplBase& heap) {
                heap.scavenge(m_deferredDecommits);
            });
        IsoHeapImplBase::finishScavenging(m_deferredDecommits);
        m_deferredDecommits.shrink(0);
    }

    if (verbose) {
        fprintf(stderr, "--after scavenging--\n");
        dumpStats();
        fprintf(stderr, "--------------------------------\n");
    }

    {
        std::unique_lock<Mutex> lock(m_mutex);
        m_lastFullScavengeTime = std::chrono::steady_clock::now();
    }
}

void Scavenger::partialScavenge()
{
    std::unique_lock<Mutex> lock(m_scavengingMutex);

    if (verbose) {
        fprintf(stderr, "--------------------------------\n");
        fprintf(stderr, "--before partial scavenging--\n");
        dumpStats();
    }

    {
        BulkDecommit decommitter;
        {
            PrintTime printTime("\npartialScavenge under lock time");
            std::lock_guard<Mutex> lock(Heap::mutex());
            for (unsigned i = numHeaps; i--;) {
                if (!isActiveHeapKind(static_cast<HeapKind>(i)))
                    continue;
                Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(i);
                size_t freeableMemory = heap.freeableMemory(lock);
                if (freeableMemory < 4 * MB)
                    continue;
                heap.scavengeToHighWatermark(lock, decommitter);
            }

            decommitter.processEager();
        }

        {
            PrintTime printTime("partialScavenge lazy decommit time");
            decommitter.processLazy();
        }

        {
            PrintTime printTime("partialScavenge mark all as eligible time");
            std::lock_guard<Mutex> lock(Heap::mutex());
            for (unsigned i = numHeaps; i--;) {
                if (!isActiveHeapKind(static_cast<HeapKind>(i)))
                    continue;
                Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(i);
                heap.markAllLargeAsEligibile(lock);
            }
        }
    }

    {
        RELEASE_BASSERT(!m_deferredDecommits.size());
        PerProcess<AllIsoHeaps>::get()->forEach(
            [&] (IsoHeapImplBase& heap) {
                heap.scavengeToHighWatermark(m_deferredDecommits);
            });
        IsoHeapImplBase::finishScavenging(m_deferredDecommits);
        m_deferredDecommits.shrink(0);
    }

    if (verbose) {
        fprintf(stderr, "--after partial scavenging--\n");
        dumpStats();
        fprintf(stderr, "--------------------------------\n");
    }

    {
        std::unique_lock<Mutex> lock(m_mutex);
        m_lastPartialScavengeTime = std::chrono::steady_clock::now();
    }
}

size_t Scavenger::freeableMemory()
{
    size_t result = 0;
    {
        std::lock_guard<Mutex> lock(Heap::mutex());
        for (unsigned i = numHeaps; i--;) {
            if (!isActiveHeapKind(static_cast<HeapKind>(i)))
                continue;
            result += PerProcess<PerHeapKind<Heap>>::get()->at(i).freeableMemory(lock);
        }
    }

    PerProcess<AllIsoHeaps>::get()->forEach(
        [&] (IsoHeapImplBase& heap) {
            result += heap.freeableMemory();
        });

    return result;
}

size_t Scavenger::footprint()
{
    RELEASE_BASSERT(!PerProcess<Environment>::get()->isDebugHeapEnabled());

    size_t result = 0;
    for (unsigned i = numHeaps; i--;) {
        if (!isActiveHeapKind(static_cast<HeapKind>(i)))
            continue;
        result += PerProcess<PerHeapKind<Heap>>::get()->at(i).footprint();
    }

    PerProcess<AllIsoHeaps>::get()->forEach(
        [&] (IsoHeapImplBase& heap) {
            result += heap.footprint();
        });

    return result;
}

void Scavenger::threadEntryPoint(Scavenger* scavenger)
{
    scavenger->threadRunLoop();
}

void Scavenger::threadRunLoop()
{
    setSelfQOSClass();
#if BOS(DARWIN)
    setThreadName("JavaScriptCore bmalloc scavenger");
#else
    setThreadName("BMScavenger");
#endif
    
    // This loop ratchets downward from most active to least active state. While
    // we ratchet downward, any other thread may reset our state.
    
    // We require any state change while we are sleeping to signal to our
    // condition variable and wake us up.
    
    while (true) {
        if (m_state == State::Sleep) {
            std::unique_lock<Mutex> lock(m_mutex);
            m_condition.wait(lock, [&]() { return m_state != State::Sleep; });
        }
        
        if (m_state == State::RunSoon) {
            std::unique_lock<Mutex> lock(m_mutex);
            m_condition.wait_for(lock, std::chrono::milliseconds(m_isInMiniMode ? 200 : 2000), [&]() { return m_state != State::RunSoon; });
        }
        
        m_state = State::Sleep;
        
        setSelfQOSClass();
        
        if (verbose) {
            fprintf(stderr, "--------------------------------\n");
            fprintf(stderr, "considering running scavenger\n");
            dumpStats();
            fprintf(stderr, "--------------------------------\n");
        }

        enum class ScavengeMode {
            None,
            Partial,
            Full
        };

        size_t freeableMemory = this->freeableMemory();

        ScavengeMode scavengeMode = [&] {
            auto timeSinceLastFullScavenge = this->timeSinceLastFullScavenge();
            auto timeSinceLastPartialScavenge = this->timeSinceLastPartialScavenge();
            auto timeSinceLastScavenge = std::min(timeSinceLastPartialScavenge, timeSinceLastFullScavenge);

            if (isUnderMemoryPressure() && freeableMemory > 1 * MB && timeSinceLastScavenge > std::chrono::milliseconds(5))
                return ScavengeMode::Full;

            if (!m_isProbablyGrowing) {
                if (timeSinceLastFullScavenge < std::chrono::milliseconds(1000) && !m_isInMiniMode)
                    return ScavengeMode::Partial;
                return ScavengeMode::Full;
            }

            if (m_isInMiniMode) {
                if (timeSinceLastFullScavenge < std::chrono::milliseconds(200))
                    return ScavengeMode::Partial;
                return ScavengeMode::Full;
            }

#if BCPU(X86_64)
            auto partialScavengeInterval = std::chrono::milliseconds(12000);
#else
            auto partialScavengeInterval = std::chrono::milliseconds(8000);
#endif
            if (timeSinceLastScavenge < partialScavengeInterval) {
                // Rate limit partial scavenges.
                return ScavengeMode::None;
            }
            if (freeableMemory < 25 * MB)
                return ScavengeMode::None;
            if (5 * freeableMemory < footprint())
                return ScavengeMode::None;
            return ScavengeMode::Partial;
        }();

        m_isProbablyGrowing = false;

        switch (scavengeMode) {
        case ScavengeMode::None: {
            runSoon();
            break;
        }
        case ScavengeMode::Partial: {
            partialScavenge();
            runSoon();
            break;
        }
        case ScavengeMode::Full: {
            scavenge();
            break;
        }
        }
    }
}

void Scavenger::setThreadName(const char* name)
{
    BUNUSED(name);
#if BOS(DARWIN)
    pthread_setname_np(name);
#elif BOS(LINUX)
    // Truncate the given name since Linux limits the size of the thread name 16 including null terminator.
    std::array<char, 16> buf;
    strncpy(buf.data(), name, buf.size() - 1);
    buf[buf.size() - 1] = '\0';
    pthread_setname_np(pthread_self(), buf.data());
#endif
}

void Scavenger::setSelfQOSClass()
{
#if BOS(DARWIN)
    pthread_set_qos_class_self_np(requestedScavengerThreadQOSClass(), 0);
#endif
}

} // namespace bmalloc

