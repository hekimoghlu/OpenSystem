/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#include "CCallHelpers.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class WasmBoundsCheckValue final : public Value {
public:
    static bool accepts(Kind kind) { return kind == WasmBoundsCheck; }
    
    ~WasmBoundsCheckValue() final;

    enum class Type {
        Pinned,
        Maximum,
    };

    union Bounds {
        GPRReg pinnedSize;
        size_t maximum;
    };

    unsigned offset() const { return m_offset; }
    Type boundsType() const { return m_boundsType; }
    Bounds bounds() const { return m_bounds; }

    B3_SPECIALIZE_VALUE_FOR_FIXED_CHILDREN(1)
    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_FIXED_CHILDREN

private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    friend class Procedure;
    friend class Value;

    static Opcode opcodeFromConstructor(Origin, GPRReg, Value*, unsigned) { return WasmBoundsCheck; }
    JS_EXPORT_PRIVATE WasmBoundsCheckValue(Origin, GPRReg pinnedGPR, Value* ptr, unsigned offset);

    static Opcode opcodeFromConstructor(Origin, Value*, unsigned, size_t) { return WasmBoundsCheck; }
    JS_EXPORT_PRIVATE WasmBoundsCheckValue(Origin, Value* ptr, unsigned offset, size_t maximum);

    unsigned m_offset;
    Type m_boundsType;
    Bounds m_bounds;

};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
