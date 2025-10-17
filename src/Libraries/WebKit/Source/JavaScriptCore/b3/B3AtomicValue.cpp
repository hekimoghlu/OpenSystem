/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#include "B3AtomicValue.h"
#include "B3ValueInlines.h"

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 {

AtomicValue::~AtomicValue() = default;

void AtomicValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    out.print(comma, "width = ", m_width);
    
    MemoryValue::dumpMeta(comma, out);
}

AtomicValue::AtomicValue(AtomicValue::AtomicValueRMW, Kind kind, Origin origin, Width width, Value* operand, Value* pointer, MemoryValue::OffsetType offset, HeapRange range, HeapRange fenceRange)
    : MemoryValue(CheckedOpcode, kind, operand->type(), Two, origin, offset, range, fenceRange, operand, pointer)
    , m_width(width)
{
    ASSERT(bestType(GP, accessWidth()) == accessType());
    
    switch (kind.opcode()) {
    case AtomicXchgAdd:
    case AtomicXchgAnd:
    case AtomicXchgOr:
    case AtomicXchgSub:
    case AtomicXchgXor:
    case AtomicXchg:
        break;
    default:
        ASSERT_NOT_REACHED();
    }
}

AtomicValue::AtomicValue(AtomicValue::AtomicValueCAS, Kind kind, Origin origin, Width width, Value* expectedValue, Value* newValue, Value* pointer, MemoryValue::OffsetType offset, HeapRange range, HeapRange fenceRange)
    : MemoryValue(CheckedOpcode, kind, kind.opcode() == AtomicWeakCAS ? Int32 : expectedValue->type(), Three, origin, offset, range, fenceRange, expectedValue, newValue, pointer)
    , m_width(width)
{
    ASSERT(bestType(GP, accessWidth()) == accessType());

    switch (kind.opcode()) {
    case AtomicWeakCAS:
    case AtomicStrongCAS:
        break;
    default:
        ASSERT_NOT_REACHED();
    }
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

