/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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

class JS_EXPORT_PRIVATE Const32Value : public Value {
public:
    static bool accepts(Kind kind) { return kind == Const32; }
    
    ~Const32Value() override;
    
    int32_t value() const { return m_value; }

    Value* negConstant(Procedure&) const override;
    Value* addConstant(Procedure&, int32_t other) const override;
    Value* addConstant(Procedure&, const Value* other) const override;
    Value* subConstant(Procedure&, const Value* other) const override;
    Value* mulConstant(Procedure&, const Value* other) const override;
    Value* checkAddConstant(Procedure&, const Value* other) const override;
    Value* checkSubConstant(Procedure&, const Value* other) const override;
    Value* checkMulConstant(Procedure&, const Value* other) const override;
    Value* checkNegConstant(Procedure&) const override;
    Value* divConstant(Procedure&, const Value* other) const override;
    Value* uDivConstant(Procedure&, const Value* other) const override;
    Value* modConstant(Procedure&, const Value* other) const override;
    Value* uModConstant(Procedure&, const Value* other) const override;
    Value* bitAndConstant(Procedure&, const Value* other) const override;
    Value* bitOrConstant(Procedure&, const Value* other) const override;
    Value* bitXorConstant(Procedure&, const Value* other) const override;
    Value* shlConstant(Procedure&, const Value* other) const override;
    Value* sShrConstant(Procedure&, const Value* other) const override;
    Value* zShrConstant(Procedure&, const Value* other) const override;
    Value* rotRConstant(Procedure&, const Value* other) const override;
    Value* rotLConstant(Procedure&, const Value* other) const override;
    Value* bitwiseCastConstant(Procedure&) const override;
    Value* iToDConstant(Procedure&) const override;
    Value* iToFConstant(Procedure&) const override;

    TriState equalConstant(const Value* other) const override;
    TriState notEqualConstant(const Value* other) const override;
    TriState lessThanConstant(const Value* other) const override;
    TriState greaterThanConstant(const Value* other) const override;
    TriState lessEqualConstant(const Value* other) const override;
    TriState greaterEqualConstant(const Value* other) const override;
    TriState aboveConstant(const Value* other) const override;
    TriState belowConstant(const Value* other) const override;
    TriState aboveEqualConstant(const Value* other) const override;
    TriState belowEqualConstant(const Value* other) const override;

    B3_SPECIALIZE_VALUE_FOR_NO_CHILDREN

protected:
    void dumpMeta(CommaPrinter&, PrintStream&) const override;

    // Protected because of ConstPtrValue
    static Opcode opcodeFromConstructor(Origin = Origin(), int32_t = 0) { return Const32; }

    Const32Value(Origin origin, int32_t value)
        : Value(CheckedOpcode, Const32, Int32, Zero, origin)
        , m_value(value)
    {
    }

private:
    friend class Procedure;
    friend class Value;

    int32_t m_value;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
