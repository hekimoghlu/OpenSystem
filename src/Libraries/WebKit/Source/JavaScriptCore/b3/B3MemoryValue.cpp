/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "B3MemoryValue.h"

#if ENABLE(B3_JIT)

#include "B3MemoryValueInlines.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

MemoryValue::~MemoryValue() = default;

bool MemoryValue::isLegalOffsetImpl(int64_t offset) const
{
    return WTF::isRepresentableAs<OffsetType>(offset) && isLegalOffset(static_cast<OffsetType>(offset));
}

Type MemoryValue::accessType() const
{
    if (isLoad())
        return type();
    // This happens to work for atomics, too. That's why AtomicValue does not need to override this.
    return child(0)->type();
}

Bank MemoryValue::accessBank() const
{
    return bankForType(accessType());
}

size_t MemoryValue::accessByteSize() const
{
    return bytesForWidth(accessWidth());
}

void MemoryValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    if (m_offset)
        out.print(comma, "offset = ", m_offset);
    if ((isLoad() && effects().reads != range())
        || (isStore() && effects().writes != range())
        || isExotic())
        out.print(comma, "range = ", range());
    if (isExotic())
        out.print(comma, "fenceRange = ", fenceRange());
}

// Use this form for Load (but not Load8Z, Load8S, or any of the Loads that have a suffix that
// describes the returned type).
MemoryValue::MemoryValue(MemoryValue::MemoryValueLoad, Kind kind, Type type, Origin origin, Value* pointer, MemoryValue::OffsetType offset, HeapRange range, HeapRange fenceRange)
    : Value(CheckedOpcode, kind, type, One, origin, pointer)
    , m_offset(offset)
    , m_range(range)
    , m_fenceRange(fenceRange)
{
    if (ASSERT_ENABLED) {
        switch (kind.opcode()) {
        case Load:
            break;
        case Load8Z:
        case Load8S:
        case Load16Z:
        case Load16S:
            ASSERT(type == Int32);
            break;
        case Store8:
        case Store16:
        case Store:
            ASSERT(type == Void);
            break;
        default:
            ASSERT_NOT_REACHED();
        }
    }
}

// Use this form for loads where the return type is implied.
MemoryValue::MemoryValue(MemoryValue::MemoryValueLoadImplied, Kind kind, Origin origin, Value* pointer, MemoryValue::OffsetType offset, HeapRange range, HeapRange fenceRange)
    : MemoryValue(kind, Int32, origin, pointer, offset, range, fenceRange)
{
    if (ASSERT_ENABLED) {
        switch (kind.opcode()) {
        case Load8Z:
        case Load8S:
        case Load16Z:
        case Load16S:
            break;
        default:
            ASSERT_NOT_REACHED();
        }
    }
}

// Use this form for stores.
MemoryValue::MemoryValue(MemoryValue::MemoryValueStore, Kind kind, Origin origin, Value* value, Value* pointer, MemoryValue::OffsetType offset, HeapRange range, HeapRange fenceRange)
    : Value(CheckedOpcode, kind, Void, Two, origin, value, pointer)
    , m_offset(offset)
    , m_range(range)
    , m_fenceRange(fenceRange)
{
    ASSERT(B3::isStore(kind.opcode()));
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
