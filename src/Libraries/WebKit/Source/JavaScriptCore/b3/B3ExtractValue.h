/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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

#include "B3Value.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class JS_EXPORT_PRIVATE ExtractValue final : public Value {
public:
#if CPU(ARM_THUMB2)
    // See LowerInt64 for details
    static constexpr int s_int64HighBits = 1;
    static constexpr int s_int64LowBits = 0;
#endif

    static bool accepts(Kind kind) { return kind == Extract; }

    ~ExtractValue() final;

    int32_t index() const { return m_index; }

    B3_SPECIALIZE_VALUE_FOR_FIXED_CHILDREN(1)
    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_FIXED_CHILDREN

private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    static Opcode opcodeFromConstructor(Origin, Type, Value*, int32_t) { return Extract; }

    ExtractValue(Origin origin, Type type, Value* tuple, int32_t index)
        : Value(CheckedOpcode, Extract, type, One, origin, tuple)
        , m_index(index)
    {
    }

    friend class Procedure;
    friend class Value;

    int32_t m_index;
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
