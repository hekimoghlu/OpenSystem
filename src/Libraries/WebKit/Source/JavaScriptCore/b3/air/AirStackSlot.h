/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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

#if ENABLE(B3_JIT)

#include "AirStackSlotKind.h"
#include "B3SparseCollection.h"
#include <limits.h>
#include <wtf/FastMalloc.h>
#include <wtf/Noncopyable.h>
#include <wtf/PrintStream.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 {

class StackSlot;

namespace Air {

class StackSlot {
    WTF_MAKE_NONCOPYABLE(StackSlot);
    WTF_MAKE_TZONE_ALLOCATED(StackSlot);
public:
    unsigned byteSize() const { return m_byteSize; }
    StackSlotKind kind() const { return m_kind; }
    bool isLocked() const { return m_kind == StackSlotKind::Locked; }
    bool isSpill() const { return m_kind == StackSlotKind::Spill; }
    unsigned index() const { return m_index; }

    void ensureSize(uint64_t requestedSize)
    {
        ASSERT(!m_offsetFromFP);
        RELEASE_ASSERT(requestedSize <= std::numeric_limits<uint32_t>::max());
        m_byteSize = std::max(m_byteSize, static_cast<uint32_t>(requestedSize));
    }

    unsigned alignment() const
    {
        if (byteSize() <= 1)
            return 1;
        if (byteSize() <= 2)
            return 2;
        if (byteSize() <= 4)
            return 4;
        return 8;
    }

    // Zero means that it's not yet assigned.
    intptr_t offsetFromFP() const { return m_offsetFromFP; }

    // This should usually just be called from phases that do stack allocation. But you can
    // totally force a stack slot to land at some offset.
    void setOffsetFromFP(intptr_t);
    
    // This computes a hash for comparing this to JSAir's StackSlot.
    unsigned jsHash() const;

    void dump(PrintStream&) const;
    void deepDump(PrintStream&) const;

private:
    friend class Code;
    friend class SparseCollection<StackSlot>;

    StackSlot(uint64_t byteSize, StackSlotKind, intptr_t offsetFromFP = 0);
    
    uint32_t m_byteSize { 0 };
    StackSlotKind m_kind { StackSlotKind::Locked };
    unsigned m_index { UINT_MAX };
    intptr_t m_offsetFromFP { 0 };
};

class DeepStackSlotDump {
public:
    DeepStackSlotDump(const StackSlot* slot)
        : m_slot(slot)
    {
    }

    void dump(PrintStream& out) const
    {
        if (m_slot)
            m_slot->deepDump(out);
        else
            out.print("<null>");
    }

private:
    const StackSlot* m_slot;
};

inline DeepStackSlotDump deepDump(const StackSlot* slot)
{
    return DeepStackSlotDump(slot);
}

} } } // namespace JSC::B3::Air

namespace WTF {

inline void printInternal(PrintStream& out, JSC::B3::Air::StackSlot* stackSlot)
{
    out.print(pointerDump(stackSlot));
}

} // namespace WTF

#endif // ENABLE(B3_JIT)
