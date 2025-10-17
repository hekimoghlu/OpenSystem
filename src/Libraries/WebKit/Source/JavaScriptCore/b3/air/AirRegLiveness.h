/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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

#include "AirBasicBlock.h"
#include "AirCode.h"
#include "AirInst.h"
#include "AirLiveness.h"
#include "RegisterSet.h"
#include <wtf/IndexMap.h>

namespace JSC { namespace B3 { namespace Air {

// Although we could trivially adapt Air::Liveness<> to work with Reg, this would not be so
// efficient. There is a small number of registers, so it's much better to use bitvectors for
// register liveness. This is a specialization of Liveness<> that uses bitvectors directly.
// This makes the code sufficiently different that it didn't make sense to try to share code.
class RegLiveness {
    struct Actions {
        Actions() = default;
        
        RegisterSet use;
        RegisterSet def;
    };
    
    typedef Vector<Actions, 0, UnsafeVectorOverflow> ActionsForBoundary;
    
public:
    typedef Reg Thing;
    
    RegLiveness(Code& code);
    ~RegLiveness();
    
    class LocalCalcBase {
    public:
        LocalCalcBase(BasicBlock* block)
            : m_block(block)
        {
        }
        
        const RegisterSet& live() const
        {
            return m_workset;
        }
        
        bool isLive(Reg reg) const
        {
            return m_workset.contains(reg, IgnoreVectors);
        }
        
    protected:
        BasicBlock* m_block;
        RegisterSet m_workset;
    };
    
    // This calculator has to be run in reverse.
    class LocalCalc : public LocalCalcBase {
    public:
        LocalCalc(RegLiveness& liveness, BasicBlock* block)
            : LocalCalcBase(block)
            , m_actions(liveness.m_actions[block])
        {
            m_workset = liveness.m_liveAtTail[block];
        }
        
        void execute(unsigned instIndex)
        {
            m_actions[instIndex + 1].def.forEach([&] (Reg r) {
                m_workset.remove(r);
            });
            m_workset.merge(m_actions[instIndex].use);
        }
        
    private:
        friend class RegLiveness;
        
        ActionsForBoundary& m_actions;
    };
    
    class LocalCalcForUnifiedTmpLiveness : public LocalCalcBase {
    public:
        LocalCalcForUnifiedTmpLiveness(UnifiedTmpLiveness& liveness, BasicBlock* block);
        
        void execute(unsigned instIndex);
        
    private:
        Code& m_code;
        UnifiedTmpLiveness::ActionsForBoundary& m_actions;
    };
    
    const RegisterSet& liveAtHead(BasicBlock* block) const
    {
        return m_liveAtHead[block];
    }
    
    const RegisterSet& liveAtTail(BasicBlock* block) const
    {
        return m_liveAtTail[block];
    }
    
private:
    IndexMap<BasicBlock*, RegisterSet> m_liveAtHead;
    IndexMap<BasicBlock*, RegisterSet> m_liveAtTail;
    IndexMap<BasicBlock*, ActionsForBoundary> m_actions;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

