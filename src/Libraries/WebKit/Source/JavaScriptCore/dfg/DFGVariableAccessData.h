/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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

#if ENABLE(DFG_JIT)

#include "DFGCommon.h"
#include "DFGDoubleFormatState.h"
#include "DFGFlushFormat.h"
#include "DFGFlushedAt.h"
#include "DFGNodeFlags.h"
#include "Operands.h"
#include "SpeculatedType.h"
#include "VirtualRegister.h"
#include <wtf/UnionFind.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace DFG {

struct Node;

enum DoubleBallot { VoteValue, VoteDouble };

class VariableAccessData : public UnionFind<VariableAccessData> {
    WTF_MAKE_TZONE_ALLOCATED(VariableAccessData);
    WTF_MAKE_NONCOPYABLE(VariableAccessData);
public:
    VariableAccessData();
    VariableAccessData(Operand);
    
    Operand operand()
    {
        ASSERT(m_operand == find()->m_operand);
        return m_operand;
    }
    
    VirtualRegister& machineLocal()
    {
        ASSERT(find() == this);
        return m_machineLocal;
    }

    bool mergeIsProfitableToUnbox(bool isProfitableToUnbox)
    {
        return checkAndSet(m_isProfitableToUnbox, m_isProfitableToUnbox || isProfitableToUnbox);
    }
    
    bool isProfitableToUnbox()
    {
        return m_isProfitableToUnbox;
    }
    
    bool mergeShouldNeverUnbox(bool shouldNeverUnbox);
    
    // Returns true if it would be unsound to store the value in an unboxed fashion.
    // If this returns false, it simply means that it is sound to unbox; it doesn't
    // mean that we have actually done so.
    bool shouldNeverUnbox()
    {
        return m_shouldNeverUnbox;
    }
    
    // Returns true if we should be unboxing the value provided that the predictions
    // and double format vote say so. This may return false even if shouldNeverUnbox()
    // returns false, since this incorporates heuristics of profitability.
    bool shouldUnboxIfPossible()
    {
        return !shouldNeverUnbox() && isProfitableToUnbox();
    }

    bool mergeStructureCheckHoistingFailed(bool failed)
    {
        return checkAndSet(m_structureCheckHoistingFailed, m_structureCheckHoistingFailed || failed);
    }
    
    bool mergeCheckArrayHoistingFailed(bool failed)
    {
        return checkAndSet(m_checkArrayHoistingFailed, m_checkArrayHoistingFailed || failed);
    }
    
    bool structureCheckHoistingFailed()
    {
        return m_structureCheckHoistingFailed;
    }
    
    bool checkArrayHoistingFailed()
    {
        return m_checkArrayHoistingFailed;
    }
    
    bool mergeIsLoadedFrom(bool isLoadedFrom)
    {
        return checkAndSet(m_isLoadedFrom, m_isLoadedFrom || isLoadedFrom);
    }
    
    void setIsLoadedFrom(bool isLoadedFrom)
    {
        m_isLoadedFrom = isLoadedFrom;
    }
    
    bool isLoadedFrom()
    {
        return m_isLoadedFrom;
    }
    
    bool predict(SpeculatedType prediction);
    
    SpeculatedType nonUnifiedPrediction()
    {
        return m_prediction;
    }
    
    SpeculatedType prediction()
    {
        return find()->m_prediction;
    }
    
    SpeculatedType argumentAwarePrediction()
    {
        return find()->m_argumentAwarePrediction;
    }
    
    bool mergeArgumentAwarePrediction(SpeculatedType prediction);
    
    void clearVotes()
    {
        ASSERT(find() == this);
        m_votes[0] = 0;
        m_votes[1] = 0;
    }
    
    void vote(unsigned ballot, float weight = 1)
    {
        ASSERT(ballot < 2);
        m_votes[ballot] += weight;
    }
    
    double voteRatio()
    {
        ASSERT(find() == this);
        return static_cast<double>(m_votes[1]) / m_votes[0];
    }
    
    bool shouldUseDoubleFormatAccordingToVote();
    
    DoubleFormatState doubleFormatState()
    {
        return find()->m_doubleFormatState;
    }
    
    bool shouldUseDoubleFormat()
    {
        ASSERT(isRoot());
        bool doubleState = m_doubleFormatState == UsingDoubleFormat;
        ASSERT(!(doubleState && shouldNeverUnbox()));
        return doubleState && isProfitableToUnbox();
    }
    
    bool tallyVotesForShouldUseDoubleFormat();
    
    bool mergeDoubleFormatState(DoubleFormatState);
    
    bool makePredictionForDoubleFormat();
    
    NodeFlags flags() const { return m_flags; }
    
    bool mergeFlags(NodeFlags newFlags)
    {
        return checkAndSet(m_flags, m_flags | newFlags);
    }
    
    FlushFormat flushFormat();
    
    bool couldRepresentInt52();
    
    FlushedAt flushedAt()
    {
        return FlushedAt(flushFormat(), machineLocal());
    }

private:
    bool couldRepresentInt52Impl();
    
    // This is slightly space-inefficient, since anything we're unified with
    // will have the same operand and should have the same prediction. But
    // putting them here simplifies the code, and we don't expect DFG space
    // usage for variable access nodes do be significant.

    SpeculatedType m_prediction;
    SpeculatedType m_argumentAwarePrediction;
    Operand m_operand;
    VirtualRegister m_machineLocal;
    NodeFlags m_flags;

    bool m_shouldNeverUnbox;
    bool m_structureCheckHoistingFailed;
    bool m_checkArrayHoistingFailed;
    bool m_isProfitableToUnbox;
    bool m_isLoadedFrom;

    float m_votes[2]; // Used primarily for double voting but may be reused for other purposes.
    DoubleFormatState m_doubleFormatState;
};

} } // namespace JSC::DFG

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(DFG_JIT)
