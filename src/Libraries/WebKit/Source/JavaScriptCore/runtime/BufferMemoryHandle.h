/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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

#include "MemoryMode.h"
#include "Options.h"
#include "PageCount.h"

#include <wtf/CagedPtr.h>
#include <wtf/Expected.h>
#include <wtf/Function.h>
#include <wtf/RAMSize.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/StdSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WTF {
class PrintStream;
}

namespace JSC {

class LLIntOffsetsExtractor;

enum class GrowFailReason : uint8_t {
    InvalidDelta,
    InvalidGrowSize,
    WouldExceedMaximum,
    OutOfMemory,
    GrowSharedUnavailable,
};

struct BufferMemoryResult {
    enum Kind {
        Success,
        SuccessAndNotifyMemoryPressure,
        SyncTryToReclaimMemory
    };

    BufferMemoryResult() { }

    BufferMemoryResult(void* basePtr, Kind kind)
        : basePtr(basePtr)
        , kind(kind)
    {
    }

    void dump(PrintStream&) const;

    void* basePtr;
    Kind kind;
};

class BufferMemoryManager {
    WTF_MAKE_TZONE_ALLOCATED(BufferMemoryManager);
    WTF_MAKE_NONCOPYABLE(BufferMemoryManager);
public:
    friend class LazyNeverDestroyed<BufferMemoryManager>;

    BufferMemoryResult tryAllocateFastMemory();
    void freeFastMemory(void* basePtr);

    BufferMemoryResult tryAllocateGrowableBoundsCheckingMemory(size_t mappedCapacity);

    void freeGrowableBoundsCheckingMemory(void* basePtr, size_t mappedCapacity);

    bool isInGrowableOrFastMemory(void* address);

    // We allow people to "commit" more wasm memory than there is on the system since most of the time
    // people don't actually write to most of that memory. There is some chance that this gets us
    // jettisoned but that's possible anyway.
    inline size_t memoryLimit() const
    {
        if (productOverflows<size_t>(ramSize(),  3))
            return std::numeric_limits<size_t>::max();
        return ramSize() * 3;
    }

    // FIXME: Ideally, bmalloc would have this kind of mechanism. Then, we would just forward to that
    // mechanism here.
    BufferMemoryResult::Kind tryAllocatePhysicalBytes(size_t bytes);

    void freePhysicalBytes(size_t bytes);

    void dump(PrintStream& out) const;

    static BufferMemoryManager& singleton();

private:
    BufferMemoryManager() = default;

    Lock m_lock;
    unsigned m_maxFastMemoryCount { Options::maxNumWasmFastMemories() };
    Vector<void*> m_fastMemories;
    StdSet<std::pair<uintptr_t, size_t>> m_growableBoundsCheckingMemories;
    size_t m_physicalBytes { 0 };
};

class BufferMemoryHandle final : public ThreadSafeRefCounted<BufferMemoryHandle> {
    WTF_MAKE_NONCOPYABLE(BufferMemoryHandle);
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(BufferMemoryHandle, JS_EXPORT_PRIVATE);
    friend LLIntOffsetsExtractor;
public:
    BufferMemoryHandle(void*, size_t size, size_t mappedCapacity, PageCount initial, PageCount maximum, MemorySharingMode, MemoryMode);
    JS_EXPORT_PRIVATE ~BufferMemoryHandle();

    void* memory() const;
    size_t size(std::memory_order order = std::memory_order_seq_cst) const
    {
        if (m_sharingMode == MemorySharingMode::Default)
            return m_size.load(std::memory_order_relaxed);
        return m_size.load(order);
    }

    std::span<uint8_t> mutableSpan(std::memory_order order = std::memory_order_seq_cst) { return { static_cast<uint8_t*>(memory()), size(order) }; }

    size_t mappedCapacity() const { return m_mappedCapacity; }
    PageCount initial() const { return m_initial; }
    PageCount maximum() const { return m_maximum; }
    MemorySharingMode sharingMode() const { return m_sharingMode; }
    MemoryMode mode() const { return m_mode; }
    static constexpr ptrdiff_t offsetOfSize() { return OBJECT_OFFSETOF(BufferMemoryHandle, m_size); }
    Lock& lock() { return m_lock; }

    void updateSize(size_t size, std::memory_order order = std::memory_order_seq_cst)
    {
        m_size.store(size, order);
    }

    static size_t fastMappedRedzoneBytes();
    static size_t fastMappedBytes();

    static void* nullBasePointer();

private:
    using CagedMemory = CagedPtr<Gigacage::Primitive, void>;

    Lock m_lock;
    MemorySharingMode m_sharingMode { MemorySharingMode::Default };
    MemoryMode m_mode { MemoryMode::BoundsChecking };
    CagedMemory m_memory;
    std::atomic<size_t> m_size { 0 };
    size_t m_mappedCapacity { 0 };
    PageCount m_initial;
    PageCount m_maximum;
};

} // namespace JSC
