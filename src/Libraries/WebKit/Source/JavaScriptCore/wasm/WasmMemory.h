/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

#if ENABLE(WEBASSEMBLY)

#include "ArrayBuffer.h"
#include "MemoryMode.h"
#include "PageCount.h"
#include "WeakGCSet.h"

#include <wtf/CagedPtr.h>
#include <wtf/Expected.h>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/Vector.h>

namespace WTF {
class PrintStream;
}

namespace JSC {

class LLIntOffsetsExtractor;

namespace Wasm {

class Memory final : public RefCounted<Memory> {
    WTF_MAKE_NONCOPYABLE(Memory);
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(Memory, JS_EXPORT_PRIVATE);
    friend LLIntOffsetsExtractor;
public:
    using JSWebAssemblyInstanceWeakCGSet = WeakGCSet<JSWebAssemblyInstance>;

    void dump(WTF::PrintStream&) const;

    enum NotifyPressure { NotifyPressureTag };
    enum SyncTryToReclaim { SyncTryToReclaimTag };
    enum GrowSuccess { GrowSuccessTag };

    static Ref<Memory> create(VM&);
    JS_EXPORT_PRIVATE static Ref<Memory> create(VM&, Ref<BufferMemoryHandle>&&, WTF::Function<void(GrowSuccess, PageCount, PageCount)>&& growSuccessCallback);
    JS_EXPORT_PRIVATE static Ref<Memory> create(VM&, Ref<SharedArrayBufferContents>&&, WTF::Function<void(GrowSuccess, PageCount, PageCount)>&& growSuccessCallback);
    JS_EXPORT_PRIVATE static Ref<Memory> createZeroSized(VM&, MemorySharingMode, WTF::Function<void(GrowSuccess, PageCount, PageCount)>&& growSuccessCallback);
    static RefPtr<Memory> tryCreate(VM&, PageCount initial, PageCount maximum, MemorySharingMode, std::optional<MemoryMode> desiredMemoryMode, WTF::Function<void(GrowSuccess, PageCount, PageCount)>&& growSuccessCallback);

    JS_EXPORT_PRIVATE ~Memory();

    static size_t fastMappedRedzoneBytes() { return BufferMemoryHandle::fastMappedRedzoneBytes(); }
    static size_t fastMappedBytes() { return BufferMemoryHandle::fastMappedBytes(); } // Includes redzone.

    static bool addressIsInGrowableOrFastMemory(void*);

    void* basePointer() const { return m_handle->memory(); }
    size_t size() const { return m_handle->size(); }
    size_t mappedCapacity() const { return m_handle->mappedCapacity(); }
    PageCount initial() const { return m_handle->initial(); }
    PageCount maximum() const { return m_handle->maximum(); }
    BufferMemoryHandle& handle() { return m_handle.get(); }

    MemorySharingMode sharingMode() const { return m_handle->sharingMode(); }
    MemoryMode mode() const { return m_handle->mode(); }

    Expected<PageCount, GrowFailReason> grow(VM&, PageCount);
    bool fill(uint32_t, uint8_t, uint32_t);
    bool copy(uint32_t, uint32_t, uint32_t);
    bool init(uint32_t, const uint8_t*, uint32_t);

    void registerInstance(JSWebAssemblyInstance&);

    void checkLifetime() { ASSERT(!deletionHasBegun()); }

    static constexpr ptrdiff_t offsetOfHandle() { return OBJECT_OFFSETOF(Memory, m_handle); }

    SharedArrayBufferContents* shared() const { return m_shared.get(); }

private:
    Memory(VM&);
    Memory(VM&, Ref<BufferMemoryHandle>&&, WTF::Function<void(GrowSuccess, PageCount, PageCount)>&& growSuccessCallback);
    Memory(VM&, Ref<BufferMemoryHandle>&&, Ref<SharedArrayBufferContents>&&, WTF::Function<void(GrowSuccess, PageCount, PageCount)>&& growSuccessCallback);
    Memory(VM&, PageCount initial, PageCount maximum, MemorySharingMode, WTF::Function<void(GrowSuccess, PageCount, PageCount)>&& growSuccessCallback);

    Expected<PageCount, GrowFailReason> growShared(VM&, PageCount);

    Ref<BufferMemoryHandle> m_handle;
    RefPtr<SharedArrayBufferContents> m_shared;
    WTF::Function<void(GrowSuccess, PageCount, PageCount)> m_growSuccessCallback;
    // FIXME: If/When merging this into JSWebAssemblyMemory we should just use an unconditionalFinalizer.

    JSWebAssemblyInstanceWeakCGSet m_instances;
};

} } // namespace JSC::Wasm

#else

namespace JSC { namespace Wasm {

class Memory {
public:
    static bool addressIsInGrowableOrFastMemory(void*) { return false; }
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
