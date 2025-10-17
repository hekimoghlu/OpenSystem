/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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

#include <wtf/StdLibExtras.h>

namespace JSC {

class Heap;
class HeapCell;
class PreciseAllocation;
class MarkedBlock;
class WeakSet;
class VM;

typedef uint32_t HeapVersion;

// This is how we abstract over either MarkedBlock& or PreciseAllocation&. Put things in here as you
// find need for them.

class CellContainer {
public:
    CellContainer()
        : m_encodedPointer(0)
    {
    }
    
    CellContainer(MarkedBlock& markedBlock)
        : m_encodedPointer(std::bit_cast<uintptr_t>(&markedBlock))
    {
    }
    
    CellContainer(PreciseAllocation& preciseAllocation)
        : m_encodedPointer(std::bit_cast<uintptr_t>(&preciseAllocation) | isPreciseAllocationBit)
    {
    }
    
    VM& vm() const;
    JSC::Heap* heap() const;
    
    explicit operator bool() const { return !!m_encodedPointer; }
    
    bool isMarkedBlock() const { return m_encodedPointer && !(m_encodedPointer & isPreciseAllocationBit); }
    bool isPreciseAllocation() const { return m_encodedPointer & isPreciseAllocationBit; }
    
    MarkedBlock& markedBlock() const
    {
        ASSERT(isMarkedBlock());
        return *std::bit_cast<MarkedBlock*>(m_encodedPointer);
    }
    
    PreciseAllocation& preciseAllocation() const
    {
        ASSERT(isPreciseAllocation());
        return *std::bit_cast<PreciseAllocation*>(m_encodedPointer - isPreciseAllocationBit);
    }
    
    bool areMarksStale() const;
    
    bool isMarked(HeapCell*) const;
    bool isMarked(HeapVersion markingVersion, HeapCell*) const;
    
    bool isNewlyAllocated(HeapCell*) const;
    
    void noteMarked();
    void assertValidCell(VM&, HeapCell*) const;
    
    size_t cellSize() const;
    
    WeakSet& weakSet() const;
    
private:
    static constexpr uintptr_t isPreciseAllocationBit = 1;
    uintptr_t m_encodedPointer;
};

} // namespace JSC

