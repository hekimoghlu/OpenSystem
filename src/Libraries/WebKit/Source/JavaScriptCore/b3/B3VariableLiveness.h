/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

#include "B3BasicBlock.h"
#include "B3CFG.h"
#include "B3Procedure.h"
#include "B3ValueInlines.h"
#include "B3Variable.h"
#include "B3VariableValue.h"
#include <wtf/Liveness.h>

namespace JSC { namespace B3 {

struct VariableLivenessAdapter {
    static constexpr const char* name = "VariableLiveness";
    typedef B3::CFG CFG;
    typedef Variable* Thing;
    
    VariableLivenessAdapter(Procedure& proc)
        : proc(proc)
    {
    }
    
    void prepareToCompute()
    {
    }
    
    unsigned numIndices()
    {
        return proc.variables().size();
    }
    
    static unsigned valueToIndex(Variable* var) { return var->index(); }
    Variable* indexToValue(unsigned index) { return proc.variables()[index]; }
    
    unsigned blockSize(BasicBlock* block)
    {
        return block->size();
    }
    
    template<typename Func>
    void forEachUse(BasicBlock* block, unsigned valueBoundaryIndex, const Func& func)
    {
        // We want all of the uses that happen between valueBoundaryIndex-1 and
        // valueBoundaryIndex. Since the Get opcode is the only value that has a use and since
        // this is an early use, we only care about block[valueBoundaryIndex].
        Value* value = block->get(valueBoundaryIndex);
        if (!value)
            return;
        if (value->opcode() != Get)
            return;
        func(value->as<VariableValue>()->variable()->index());
    }
    
    template<typename Func>
    void forEachDef(BasicBlock* block, unsigned valueBoundaryIndex, const Func& func)
    {
        // We want all of the defs that happen between valueBoundaryIndex-1 and
        // valueBoundaryIndex. Since the Set opcode is the only value that has a def and since
        // this is an late def, we only care about block[valueBoundaryIndex - 1].
        Value* value = block->get(valueBoundaryIndex - 1);
        if (!value)
            return;
        if (value->opcode() != Set)
            return;
        func(value->as<VariableValue>()->variable()->index());
    }
    
    Procedure& proc;
};

class VariableLiveness : public WTF::Liveness<VariableLivenessAdapter> {
public:
    VariableLiveness(Procedure&);
    ~VariableLiveness();
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

