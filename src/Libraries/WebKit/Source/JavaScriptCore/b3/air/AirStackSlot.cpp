/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
#include "config.h"
#include "AirStackSlot.h"

#if ENABLE(B3_JIT)

#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 { namespace Air {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StackSlot);

void StackSlot::setOffsetFromFP(intptr_t value)
{
    m_offsetFromFP = value;
}

unsigned StackSlot::jsHash() const
{
    return static_cast<unsigned>(m_kind) + m_byteSize * 3 + m_offsetFromFP * 7;
}

void StackSlot::dump(PrintStream& out) const
{
    if (isSpill())
        out.print("spill");
    else
        out.print("stack");
    out.print(m_index);
}

void StackSlot::deepDump(PrintStream& out) const
{
    out.print("byteSize = ", m_byteSize, ", offsetFromFP = ", m_offsetFromFP, ", kind = ", m_kind);
}

StackSlot::StackSlot(uint64_t byteSize, StackSlotKind kind, intptr_t offsetFromFP)
    : m_byteSize(static_cast<uint32_t>(byteSize))
    , m_kind(kind)
    , m_offsetFromFP(offsetFromFP)
{
    ASSERT(byteSize);
    RELEASE_ASSERT(byteSize <= std::numeric_limits<uint32_t>::max());
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
