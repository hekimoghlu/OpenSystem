/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#include "B3EliminateDeadCode.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlock.h"
#include "B3PhaseScope.h"
#include "B3ValueInlines.h"
#include "B3Variable.h"
#include "B3VariableValue.h"
#include <wtf/GraphNodeWorklist.h>
#include <wtf/IndexSet.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

// FIXME: this pass currently only eliminates values and variables, it does not seem to eliminate dead blocks.

bool eliminateDeadCodeImpl(Procedure& proc)
{
    bool changed = false;
    GraphNodeWorklist<Value*, IndexSet<Value*>> worklist;
    Vector<UpsilonValue*, 128> upsilons;
    for (BasicBlock* block : proc) {
        for (Value* value : *block) {
            Effects effects;
            // We don't care about effects of SSA operations, since we model them more
            // accurately than the effects() method does.
            if (value->opcode() != Phi && value->opcode() != Upsilon)
                effects = value->effects();
            
            if (effects.mustExecute())
                worklist.push(value);
            
            if (UpsilonValue* upsilon = value->as<UpsilonValue>())
                upsilons.append(upsilon);
        }
    }
    for (;;) {
        while (Value* value = worklist.pop()) {
            for (Value* child : value->children())
                worklist.push(child);
        }
        
        bool didPush = false;
        for (size_t upsilonIndex = 0; upsilonIndex < upsilons.size(); ++upsilonIndex) {
            UpsilonValue* upsilon = upsilons[upsilonIndex];
            if (worklist.saw(upsilon->phi())) {
                worklist.push(upsilon);
                upsilons[upsilonIndex--] = upsilons.last();
                upsilons.takeLast();
                didPush = true;
            }
        }
        if (!didPush)
            break;
    }

    IndexSet<Variable*> liveVariables;
    
    for (BasicBlock* block : proc) {
        size_t sourceIndex = 0;
        size_t targetIndex = 0;
        while (sourceIndex < block->size()) {
            Value* value = block->at(sourceIndex++);
            if (worklist.saw(value)) {
                if (VariableValue* variableValue = value->as<VariableValue>())
                    liveVariables.add(variableValue->variable());
                block->at(targetIndex++) = value;
            } else {
                proc.deleteValue(value);
                changed = true;
            }
        }
        block->values().shrink(targetIndex);
    }

    for (Variable* variable : proc.variables()) {
        if (!liveVariables.contains(variable))
            proc.deleteVariable(variable);
    }
    return changed;
}

bool eliminateDeadCode(Procedure& proc)
{
    PhaseScope phaseScope(proc, "eliminateDeadCode"_s);
    return eliminateDeadCodeImpl(proc);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
