/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#include "Reg.h"

namespace JSC { namespace B3 {

class JS_EXPORT_PRIVATE ArgumentRegValue final : public Value {
public:
    enum VectorTag { UsesVectorArgs };

    static bool accepts(Kind kind) { return kind == ArgumentReg; }
    
    ~ArgumentRegValue() final;

    Reg argumentReg() const { return m_reg; }

    B3_SPECIALIZE_VALUE_FOR_NO_CHILDREN

private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    friend class Procedure;
    friend class Value;
    
    static Opcode opcodeFromConstructor(Origin, Reg) { return ArgumentReg; }
    static Opcode opcodeFromConstructor(Origin, Reg, VectorTag) { return ArgumentReg; }

    ArgumentRegValue(Origin origin, Reg reg)
        : Value(CheckedOpcode, ArgumentReg, reg.isGPR() ? pointerType() : Double, Zero, origin)
        , m_reg(reg)
    {
        ASSERT(reg.isSet());
    }

    ArgumentRegValue(Origin origin, Reg reg, VectorTag)
        : Value(CheckedOpcode, ArgumentReg, V128, Zero, origin)
        , m_reg(reg)
    {
        ASSERT(reg.isSet());
        ASSERT(reg.isFPR());
    }

    Reg m_reg;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
