/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#include "B3MemoryValue.h"
#include "B3Width.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class JS_EXPORT_PRIVATE AtomicValue final : public MemoryValue {
public:
    static bool accepts(Kind kind)
    {
        return isAtom(kind.opcode());
    }
    
    ~AtomicValue() final;
    
    Type accessType() const { return child(0)->type(); }
    
    Width accessWidth() const { return m_width; }

    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_FIXED_CHILDREN
    
private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    friend class Procedure;
    friend class Value;

    enum AtomicValueRMW { AtomicValueRMWTag };
    enum AtomicValueCAS { AtomicValueCASTag };

    AtomicValue(Kind kind, Origin origin, Width width, Value* operand, Value* pointer)
        : AtomicValue(kind, origin, width, operand, pointer, 0)
    {
    }
    template<typename Int,
        typename = typename std::enable_if<std::is_integral<Int>::value>::type,
        typename = typename std::enable_if<std::is_signed<Int>::value>::type,
        typename = typename std::enable_if<sizeof(Int) <= sizeof(OffsetType)>::type
    >
    AtomicValue(Kind kind, Origin origin, Width width, Value* operand, Value* pointer, Int offset, HeapRange range = HeapRange::top(), HeapRange fenceRange = HeapRange::top())
        : AtomicValue(AtomicValueRMWTag, kind, origin, width, operand, pointer, offset, range, fenceRange)
    {
    }

    AtomicValue(Kind kind, Origin origin, Width width, Value* expectedValue, Value* newValue, Value* pointer)
        : AtomicValue(kind, origin, width, expectedValue, newValue, pointer, 0)
    {
    }
    template<typename Int,
        typename = typename std::enable_if<std::is_integral<Int>::value>::type,
        typename = typename std::enable_if<std::is_signed<Int>::value>::type,
        typename = typename std::enable_if<sizeof(Int) <= sizeof(OffsetType)>::type
    >
    AtomicValue(Kind kind, Origin origin, Width width, Value* expectedValue, Value* newValue, Value* pointer, Int offset, HeapRange range = HeapRange::top(), HeapRange fenceRange = HeapRange::top())
        : AtomicValue(AtomicValueCASTag, kind, origin, width, expectedValue, newValue, pointer, offset, range, fenceRange)
    {
    }

    // The above templates forward to these implementations.
    AtomicValue(AtomicValueRMW, Kind, Origin, Width, Value* operand, Value* pointer, OffsetType, HeapRange = HeapRange::top(), HeapRange fenceRange = HeapRange::top());
    AtomicValue(AtomicValueCAS, Kind, Origin, Width, Value* expectedValue, Value* newValue, Value* pointer, OffsetType, HeapRange = HeapRange::top(), HeapRange fenceRange = HeapRange::top());

    Width m_width;
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
