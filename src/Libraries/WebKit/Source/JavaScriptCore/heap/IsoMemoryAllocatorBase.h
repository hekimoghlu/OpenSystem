/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#include "AlignedMemoryAllocator.h"
#include <wtf/BitVector.h>
#include <wtf/DebugHeap.h>
#include <wtf/HashMap.h>
#include <wtf/Vector.h>


namespace JSC {

class IsoMemoryAllocatorBase : public AlignedMemoryAllocator {
public:
    IsoMemoryAllocatorBase(CString);
    ~IsoMemoryAllocatorBase() override;

    void* tryAllocateAlignedMemory(size_t alignment, size_t size) final;
    void freeAlignedMemory(void*) final;

protected:
    void releaseMemoryFromSubclassDestructor();
    virtual void* tryMallocBlock() = 0;
    virtual void freeBlock(void* block) = 0;
    virtual void commitBlock(void* block) = 0;
    virtual void decommitBlock(void* block) = 0;

private:
    Vector<void*> m_blocks;
    UncheckedKeyHashMap<void*, unsigned> m_blockIndices;
    BitVector m_committed;
    unsigned m_firstUncommitted { 0 };
    Lock m_lock;
};

} // namespace JSC

