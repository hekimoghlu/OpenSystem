/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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

namespace JSC { namespace B3 {

class JS_EXPORT_PRIVATE Const128Value final : public Value {
public:
    static inline bool accepts(Kind kind) { return kind == Const128; }

    ~Const128Value() final;

    inline v128_t value() const { return m_value; }

    Value* vectorAndConstant(Procedure&, const Value* other) const final;
    Value* vectorOrConstant(Procedure&, const Value* other) const final;
    Value* vectorXorConstant(Procedure&, const Value* other) const final;

    B3_SPECIALIZE_VALUE_FOR_NO_CHILDREN

protected:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    inline Const128Value(Origin origin, v128_t value)
        : Value(CheckedOpcode, Const128, V128, Zero, origin)
        , m_value(value)
    {
    }

private:
    static inline Opcode opcodeFromConstructor(Origin, v128_t) { return Const128; }
    friend class Procedure;
    friend class Value;

    v128_t m_value;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
