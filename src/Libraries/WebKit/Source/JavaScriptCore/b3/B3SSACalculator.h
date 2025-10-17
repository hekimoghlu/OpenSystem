/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

#include "B3Dominators.h"
#include "B3ProcedureInlines.h"
#include <wtf/Bag.h>
#include <wtf/IndexMap.h>
#include <wtf/SegmentedVector.h>

namespace JSC { namespace B3 {

// SSACalculator provides a reusable tool for building SSA's. It's modeled after
// DFG::SSACalculator.

class SSACalculator {
public:
    SSACalculator(Procedure&);
    ~SSACalculator();

    void reset();

    class Variable {
    public:
        unsigned index() const { return m_index; }
        
        void dump(PrintStream&) const;
        void dumpVerbose(PrintStream&) const;
        
    private:
        friend class SSACalculator;
        
        Variable()
            : m_index(UINT_MAX)
        {
        }
        
        Variable(unsigned index)
            : m_index(index)
        {
        }

        Vector<BasicBlock*, 4> m_blocksWithDefs;
        unsigned m_index;
    };

    class Def {
    public:
        Variable* variable() const { return m_variable; }
        BasicBlock* block() const { return m_block; }
        
        Value* value() const { return m_value; }
        
        void dump(PrintStream&) const;
        
    private:
        friend class SSACalculator;
        
        Def()
            : m_variable(nullptr)
            , m_block(nullptr)
            , m_value(nullptr)
        {
        }
        
        Def(Variable* variable, BasicBlock* block, Value* value)
            : m_variable(variable)
            , m_block(block)
            , m_value(value)
        {
        }
        
        Variable* m_variable;
        BasicBlock* m_block;
        Value* m_value;
    };

    Variable* newVariable();
    Def* newDef(Variable*, BasicBlock*, Value*);

    Variable* variable(unsigned index) { return &m_variables[index]; }

    template<typename Functor>
    void computePhis(const Functor& functor)
    {
        m_dominators = &m_proc.dominators();
        for (Variable& variable : m_variables) {
            m_dominators->forAllBlocksInPrunedIteratedDominanceFrontierOf(
                variable.m_blocksWithDefs,
                [&] (BasicBlock* block) -> bool {
                    Value* phi = functor(&variable, block);
                    if (!phi)
                        return false;

                    BlockData& data = m_data[block];
                    Def* phiDef = m_phis.add(Def(&variable, block, phi));
                    data.m_phis.append(phiDef);

                    data.m_defs.add(&variable, phiDef);
                    return true;
                });
        }
    }

    const Vector<Def*>& phisForBlock(BasicBlock* block)
    {
        return m_data[block].m_phis;
    }
    
    // Ignores defs within the given block; it assumes that you've taken care of those
    // yourself.
    Def* nonLocalReachingDef(BasicBlock*, Variable*);
    Def* reachingDefAtHead(BasicBlock* block, Variable* variable)
    {
        return nonLocalReachingDef(block, variable);
    }
    
    // Considers the def within the given block, but only works at the tail of the block.
    Def* reachingDefAtTail(BasicBlock*, Variable*);
    
    void dump(PrintStream&) const;
    
private:
    SegmentedVector<Variable> m_variables;
    Bag<Def> m_defs;
    
    Bag<Def> m_phis;
    
    struct BlockData {
        UncheckedKeyHashMap<Variable*, Def*> m_defs;
        Vector<Def*> m_phis;
    };
    
    IndexMap<BasicBlock*, BlockData> m_data;

    Dominators* m_dominators { nullptr };
    Procedure& m_proc;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
