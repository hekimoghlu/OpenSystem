/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

class JS_EXPORT_PRIVATE ConstDoubleValue final : public Value {
public:
    static bool accepts(Kind kind) { return kind == ConstDouble; }
    
    ~ConstDoubleValue() final;
    
    double value() const { return m_value; }

    Value* negConstant(Procedure&) const final;
    Value* addConstant(Procedure&, int32_t other) const final;
    Value* addConstant(Procedure&, const Value* other) const final;
    Value* subConstant(Procedure&, const Value* other) const final;
    Value* divConstant(Procedure&, const Value* other) const final;
    Value* modConstant(Procedure&, const Value* other) const final;
    Value* mulConstant(Procedure&, const Value* other) const final;
    Value* bitAndConstant(Procedure&, const Value* other) const final;
    Value* bitOrConstant(Procedure&, const Value* other) const final;
    Value* bitXorConstant(Procedure&, const Value* other) const final;
    Value* bitwiseCastConstant(Procedure&) const final;
    Value* doubleToFloatConstant(Procedure&) const final;
    Value* absConstant(Procedure&) const final;
    Value* ceilConstant(Procedure&) const final;
    Value* floorConstant(Procedure&) const final;
    Value* sqrtConstant(Procedure&) const final;
    Value* fMinConstant(Procedure&, const Value* other) const final;
    Value* fMaxConstant(Procedure&, const Value* other) const final;

    TriState equalConstant(const Value* other) const final;
    TriState notEqualConstant(const Value* other) const final;
    TriState lessThanConstant(const Value* other) const final;
    TriState greaterThanConstant(const Value* other) const final;
    TriState lessEqualConstant(const Value* other) const final;
    TriState greaterEqualConstant(const Value* other) const final;
    TriState equalOrUnorderedConstant(const Value* other) const final;

    B3_SPECIALIZE_VALUE_FOR_NO_CHILDREN

private:
    friend class Procedure;
    friend class Value;

    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    static Opcode opcodeFromConstructor(Origin, double) { return ConstDouble; }

    ConstDoubleValue(Origin origin, double value)
        : Value(CheckedOpcode, ConstDouble, Double, Zero, origin)
        , m_value(value)
    {
    }
    
    double m_value;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
