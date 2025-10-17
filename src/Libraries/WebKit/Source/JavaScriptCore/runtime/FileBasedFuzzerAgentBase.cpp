/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include "FileBasedFuzzerAgentBase.h"

#include "CodeBlock.h"
#include "JSCellInlines.h"
#include <wtf/text/MakeString.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

FileBasedFuzzerAgentBase::FileBasedFuzzerAgentBase(VM&)
{
}

String FileBasedFuzzerAgentBase::createLookupKey(const String& sourceFilename, OpcodeID opcodeId, int startLocation, int endLocation)
{
    return makeString(sourceFilename, '|',
        opcodeNames[opcodeAliasForLookupKey(opcodeId)],
        '|', startLocation, '|', endLocation);
}

OpcodeID FileBasedFuzzerAgentBase::opcodeAliasForLookupKey(const OpcodeID& opcodeId)
{
    if (opcodeId == op_call_varargs || opcodeId == op_call_direct_eval || opcodeId == op_tail_call || opcodeId == op_tail_call_varargs)
        return op_call;
    if (opcodeId == op_enumerator_get_by_val || opcodeId == op_get_by_val_with_this)
        return op_get_by_val;
    if (opcodeId == op_construct_varargs)
        return op_construct;
    if (opcodeId == op_super_construct_varargs)
        return op_super_construct;
    return opcodeId;
}

SpeculatedType FileBasedFuzzerAgentBase::getPrediction(CodeBlock* codeBlock, const CodeOrigin& codeOrigin, SpeculatedType original)
{
    Locker locker { m_lock };

    ScriptExecutable* ownerExecutable = codeBlock->ownerExecutable();
    const auto& sourceURL = ownerExecutable->sourceURL();
    if (sourceURL.isEmpty())
        return original;

    PredictionTarget predictionTarget;
    BytecodeIndex bytecodeIndex = codeOrigin.bytecodeIndex();
    predictionTarget.info = codeBlock->expressionInfoForBytecodeIndex(bytecodeIndex);

    Vector<String> urlParts = sourceURL.split('/');
    predictionTarget.sourceFilename = urlParts.isEmpty() ? sourceURL : urlParts.last();

    const auto& instructions = codeBlock->instructions();
    const auto* anInstruction = instructions.at(bytecodeIndex).ptr();
    predictionTarget.opcodeId = anInstruction->opcodeID();

    int startLocation = predictionTarget.info.divot - predictionTarget.info.startOffset;
    int endLocation = predictionTarget.info.divot + predictionTarget.info.endOffset;
    predictionTarget.lookupKey = createLookupKey(predictionTarget.sourceFilename, predictionTarget.opcodeId, startLocation, endLocation);
    return getPredictionInternal(codeBlock, predictionTarget, original);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
