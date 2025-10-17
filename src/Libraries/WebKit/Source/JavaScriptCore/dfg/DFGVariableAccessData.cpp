/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include "DFGVariableAccessData.h"

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

VariableAccessData::VariableAccessData()
    : m_prediction(SpecNone)
    , m_argumentAwarePrediction(SpecNone)
    , m_flags(0)
    , m_shouldNeverUnbox(false)
    , m_structureCheckHoistingFailed(false)
    , m_checkArrayHoistingFailed(false)
    , m_isProfitableToUnbox(false)
    , m_isLoadedFrom(false)
    , m_doubleFormatState(EmptyDoubleFormatState)
{
    clearVotes();
}

VariableAccessData::VariableAccessData(Operand operand)
    : m_prediction(SpecNone)
    , m_argumentAwarePrediction(SpecNone)
    , m_operand(operand)
    , m_flags(0)
    , m_shouldNeverUnbox(false)
    , m_structureCheckHoistingFailed(false)
    , m_checkArrayHoistingFailed(false)
    , m_isProfitableToUnbox(false)
    , m_isLoadedFrom(false)
    , m_doubleFormatState(EmptyDoubleFormatState)
{
    clearVotes();
}

bool VariableAccessData::mergeShouldNeverUnbox(bool shouldNeverUnbox)
{
    bool newShouldNeverUnbox = m_shouldNeverUnbox | shouldNeverUnbox;
    if (newShouldNeverUnbox == m_shouldNeverUnbox)
        return false;
    m_shouldNeverUnbox = newShouldNeverUnbox;
    return true;
}

bool VariableAccessData::predict(SpeculatedType prediction)
{
    VariableAccessData* self = find();
    bool result = mergeSpeculation(self->m_prediction, prediction);
    if (result)
        mergeSpeculation(m_argumentAwarePrediction, m_prediction);
    return result;
}

bool VariableAccessData::mergeArgumentAwarePrediction(SpeculatedType prediction)
{
    return mergeSpeculation(find()->m_argumentAwarePrediction, prediction);
}

bool VariableAccessData::shouldUseDoubleFormatAccordingToVote()
{
    // We don't support this facility for arguments, yet.
    // FIXME: make this work for arguments.
    if (operand().isArgument())
        return false;
        
    // If the variable is not a number prediction, then this doesn't
    // make any sense.
    if (!isFullNumberSpeculation(prediction())) {
        // FIXME: we may end up forcing a local in inlined argument position to be a double even
        // if it is sometimes not even numeric, since this never signals the fact that it doesn't
        // want doubles. https://bugs.webkit.org/show_bug.cgi?id=109511
        return false;
    }
        
    // If the variable is predicted to hold only doubles, then it's a
    // no-brainer: it should be formatted as a double.
    if (isDoubleSpeculation(prediction()))
        return true;
        
    // If the variable is known to be used as an integer, then be safe -
    // don't force it to be a double.
    if (flags() & NodeBytecodeUsesAsInt)
        return false;
        
    // If the variable has been voted to become a double, then make it a
    // double.
    if (voteRatio() >= Options::doubleVoteRatioForDoubleFormat())
        return true;
        
    return false;
}

bool VariableAccessData::tallyVotesForShouldUseDoubleFormat()
{
    ASSERT(isRoot());
        
    if (operand().isArgument() || shouldNeverUnbox()
        || (flags() & NodeBytecodePrefersArrayIndex))
        return DFG::mergeDoubleFormatState(m_doubleFormatState, NotUsingDoubleFormat);
    
    if (m_doubleFormatState == CantUseDoubleFormat)
        return false;
        
    bool newValueOfShouldUseDoubleFormat = shouldUseDoubleFormatAccordingToVote();
    if (!newValueOfShouldUseDoubleFormat) {
        // We monotonically convert to double. Hence, if the fixpoint leads us to conclude that we should
        // switch back to int, we instead ignore this and stick with double.
        return DFG::mergeDoubleFormatState(m_doubleFormatState, NotUsingDoubleFormat);
    }
        
    if (m_doubleFormatState == UsingDoubleFormat)
        return false;
        
    return DFG::mergeDoubleFormatState(m_doubleFormatState, UsingDoubleFormat);
}

bool VariableAccessData::mergeDoubleFormatState(DoubleFormatState doubleFormatState)
{
    return DFG::mergeDoubleFormatState(find()->m_doubleFormatState, doubleFormatState);
}

bool VariableAccessData::makePredictionForDoubleFormat()
{
    ASSERT(isRoot());
    
    if (m_doubleFormatState != UsingDoubleFormat)
        return false;
    
    SpeculatedType type = m_prediction;
    if (type & ~SpecBytecodeNumber)
        type |= SpecDoublePureNaN;
    if (type & (SpecInt32Only | SpecInt52Any))
        type |= SpecAnyIntAsDouble;
    return checkAndSet(m_prediction, type);
}

bool VariableAccessData::couldRepresentInt52()
{
    if (shouldNeverUnbox())
        return false;
    
    return couldRepresentInt52Impl();
}

bool VariableAccessData::couldRepresentInt52Impl()
{
    // The hardware has to support it.
    if (!enableInt52())
        return false;
    
    // We punt for machine arguments.
    if (operand().isArgument())
        return false;
    
    // The argument-aware prediction -- which merges all of an (inlined or machine)
    // argument's variable access datas' predictions -- must possibly be Int52Any.
    return isInt32OrInt52Speculation(argumentAwarePrediction());
}

FlushFormat VariableAccessData::flushFormat()
{
    ASSERT(find() == this);
    
    if (!shouldUnboxIfPossible())
        return FlushedJSValue;
    
    if (shouldUseDoubleFormat())
        return FlushedDouble;
    
    SpeculatedType prediction = argumentAwarePrediction();
    
    // This guard is here to protect the call to couldRepresentInt52(), which will return
    // true for !prediction.
    if (!prediction)
        return FlushedJSValue;
    
    if (isInt32Speculation(prediction))
        return FlushedInt32;
    
    if (couldRepresentInt52Impl())
        return FlushedInt52;
    
    if (isCellSpeculation(prediction))
        return FlushedCell;
    
    if (isBooleanSpeculation(prediction))
        return FlushedBoolean;
    
    return FlushedJSValue;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

