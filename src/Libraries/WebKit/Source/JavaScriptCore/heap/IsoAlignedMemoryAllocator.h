/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#include "IsoMemoryAllocatorBase.h"
#include <wtf/BitVector.h>
#include <wtf/HashMap.h>
#include <wtf/Vector.h>

#if ENABLE(MALLOC_HEAP_BREAKDOWN)
#include <wtf/DebugHeap.h>
#endif

namespace JSC {

class IsoAlignedMemoryAllocator final : public IsoMemoryAllocatorBase {
public:
    using Base = IsoMemoryAllocatorBase;

    IsoAlignedMemoryAllocator(CString);
    ~IsoAlignedMemoryAllocator() final;

    void dump(PrintStream&) const final;

    void* tryAllocateMemory(size_t) final;
    void freeMemory(void*) final;
    void* tryReallocateMemory(void*, size_t) final;

protected:
    void* tryMallocBlock() final;
    void freeBlock(void* block) final;
    void commitBlock(void* block) final;
    void decommitBlock(void* block) final;

#if ENABLE(MALLOC_HEAP_BREAKDOWN)
    WTF::DebugHeap m_heap;
#endif
};

} // namespace JSC

