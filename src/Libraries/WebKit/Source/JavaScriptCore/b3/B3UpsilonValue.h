/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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

class JS_EXPORT_PRIVATE UpsilonValue final : public Value {
public:
    static bool accepts(Kind kind) { return kind == Upsilon; }

    ~UpsilonValue() final;

    Value* phi() const { return m_phi; }
    void setPhi(Value* phi)
    {
        ASSERT(child(0)->type() == phi->type());
        ASSERT(phi->opcode() == Phi);
        m_phi = phi;
    }

    B3_SPECIALIZE_VALUE_FOR_FIXED_CHILDREN(1)
    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_FIXED_CHILDREN

private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    friend class Procedure;
    friend class Value;

    static Opcode opcodeFromConstructor(Origin, Value*, Value* = nullptr) { return Upsilon; }
    // Note that passing the Phi during construction is optional. A valid pattern is to first create
    // the Upsilons without the Phi, then create the Phi, then go back and tell the Upsilons about
    // the Phi. This allows you to emit code in its natural order.
    UpsilonValue(Origin origin, Value* value, Value* phi = nullptr)
        : Value(CheckedOpcode, Upsilon, Void, One, origin, value)
        , m_phi(phi)
    {
        if (phi)
            ASSERT(value->type() == phi->type());
    }

    Value* m_phi;
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
