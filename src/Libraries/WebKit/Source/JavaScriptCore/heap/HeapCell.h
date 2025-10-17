/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

#include "DestructionMode.h"
#include "EnsureStillAliveHere.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class CellContainer;
class Heap;
class PreciseAllocation;
class MarkedBlock;
class Subspace;
class VM;
struct CellAttributes;

class HeapCell {
public:
    enum Kind : int8_t {
        JSCell,
        JSCellWithIndexingHeader,
        Auxiliary
    };
    
    HeapCell() { }
    
    // We're intentionally only zapping the bits for the structureID and leaving
    // the rest of the cell header bits intact for crash analysis uses.
    enum ZapReason : int8_t { Unspecified, Destruction, StopAllocating };
    void zap(ZapReason reason)
    {
        uint32_t* cellWords = std::bit_cast<uint32_t*>(this);
        cellWords[0] = 0;
        // Leaving cellWords[1] alone for crash analysis if needed.
        cellWords[2] = reason;
    }
    bool isZapped() const { return !*std::bit_cast<const uint32_t*>(this); }

    void notifyNeedsDestruction() const;

    // isPendingDestruction returns true iff the cell is no longer alive but has not yet
    // been swept and therefore its destructor (if it has one) has not yet run.
    bool isPendingDestruction();

    bool isPreciseAllocation() const;
    CellContainer cellContainer() const;
    MarkedBlock& markedBlock() const;
    PreciseAllocation& preciseAllocation() const;

    // If you want performance and you know that your cell is small, you can do this instead:
    // ASSERT(!cell->isPreciseAllocation());
    // cell->markedBlock().vm()
    // We currently only use this hack for callees to make CallFrame::vm() fast. It's not
    // recommended to use it for too many other things, since the large allocation cutoff is
    // a runtime option and its default value is small (400 bytes).
    JSC::Heap* heap() const;
    VM& vm() const;
    
    size_t cellSize() const;
    CellAttributes cellAttributes() const;
    DestructionMode destructionMode() const;
    Kind cellKind() const;
    Subspace* subspace() const;
    
    // Call use() after the last point where you need `this` pointer to be kept alive. You usually don't
    // need to use this, but it might be necessary if you're otherwise referring to an object's innards
    // but not the object itself.
    ALWAYS_INLINE void use() const
    {
        ensureStillAliveHere(this);
    }
};

inline bool isJSCellKind(HeapCell::Kind kind)
{
    return kind == HeapCell::JSCell || kind == HeapCell::JSCellWithIndexingHeader;
}

inline bool mayHaveIndexingHeader(HeapCell::Kind kind)
{
    return kind == HeapCell::Auxiliary || kind == HeapCell::JSCellWithIndexingHeader;
}

} // namespace JSC

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::HeapCell::Kind);

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
