/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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

#include "JSExportMacros.h"

#if ENABLE(LIBPAS_JIT_HEAP) && ENABLE(JIT)
#include <wtf/CodePtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#else
#include <wtf/MetaAllocatorHandle.h>
#endif

#if !USE(SYSTEM_MALLOC)
#include <bmalloc/BPlatform.h>
#if BENABLE(LIBPAS) && (OS(DARWIN) || OS(LINUX))
#define ENABLE_LIBPAS_JIT_HEAP 1
#endif
#endif

namespace JSC {

#if ENABLE(LIBPAS_JIT_HEAP) && ENABLE(JIT)
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ExecutableMemoryHandle);
class ExecutableMemoryHandle : public ThreadSafeRefCounted<ExecutableMemoryHandle> {
    WTF_MAKE_FAST_COMPACT_ALLOCATED_WITH_HEAP_IDENTIFIER(ExecutableMemoryHandle);

public:
    using MemoryPtr = CodePtr<WTF::HandleMemoryPtrTag>;

    // Don't call this directly - for proper accounting it's necessary to call
    // ExecutableAllocator::allocate().
    JS_EXPORT_PRIVATE static RefPtr<ExecutableMemoryHandle> createImpl(size_t sizeInBytes);

    JS_EXPORT_PRIVATE ~ExecutableMemoryHandle();

    MemoryPtr start() const
    {
        return m_start;
    }

    MemoryPtr end() const
    {
        return MemoryPtr::fromUntaggedPtr(reinterpret_cast<void*>(endAsInteger()));
    }

    uintptr_t startAsInteger() const
    {
        return m_start.untaggedPtr<uintptr_t>();
    }

    uintptr_t endAsInteger() const
    {
        return startAsInteger() + sizeInBytes();
    }

    size_t sizeInBytes() const { return m_sizeInBytes; }

    bool containsIntegerAddress(uintptr_t address) const
    {
        uintptr_t startAddress = startAsInteger();
        uintptr_t endAddress = startAddress + sizeInBytes();
        return address >= startAddress && address < endAddress;
    }

    bool contains(void* address) const
    {
        return containsIntegerAddress(reinterpret_cast<uintptr_t>(address));
    }

    JS_EXPORT_PRIVATE void shrink(size_t newSizeInBytes);

    void* key() const
    {
        return m_start.untaggedPtr();
    }

    void dump(PrintStream& out) const
    {
        out.print(RawPointer(key()));
    }

private:
    ExecutableMemoryHandle(MemoryPtr start, size_t sizeInBytes)
        : m_sizeInBytes(sizeInBytes)
        , m_start(start)
    {
        ASSERT(sizeInBytes == m_sizeInBytes); // executable memory region does not exceed 4GB.
    }

    unsigned m_sizeInBytes;
    MemoryPtr m_start;
};
#else // not (ENABLE(LIBPAS_JIT_HEAP) && ENABLE(JIT))
typedef WTF::MetaAllocatorHandle ExecutableMemoryHandle;
#endif // ENABLE(LIBPAS_JIT_HEAP) && ENABLE(JIT)

} // namespace JSC

