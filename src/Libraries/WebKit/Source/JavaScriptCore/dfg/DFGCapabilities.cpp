/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#include "DFGCapabilities.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGCommon.h"
#include "ExecutableBaseInlines.h"
#include "JSCellInlines.h"
#include "Options.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace DFG {

bool isSupported()
{
    return Options::useDFGJIT() && MacroAssembler::supportsFloatingPoint();
}

bool isSupportedForInlining(CodeBlock* codeBlock)
{
    return codeBlock->ownerExecutable()->isInliningCandidate();
}

bool mightCompileEval(CodeBlock* codeBlock)
{
    return isSupported()
        && codeBlock->bytecodeCost() <= Options::maximumOptimizationCandidateBytecodeCost()
        && codeBlock->ownerExecutable()->isOkToOptimize();
}
bool mightCompileProgram(CodeBlock* codeBlock)
{
    return isSupported()
        && codeBlock->bytecodeCost() <= Options::maximumOptimizationCandidateBytecodeCost()
        && codeBlock->ownerExecutable()->isOkToOptimize();
}
bool mightCompileFunctionForCall(CodeBlock* codeBlock)
{
    return isSupported()
        && codeBlock->bytecodeCost() <= Options::maximumOptimizationCandidateBytecodeCost()
        && codeBlock->ownerExecutable()->isOkToOptimize();
}
bool mightCompileFunctionForConstruct(CodeBlock* codeBlock)
{
    return isSupported()
        && codeBlock->bytecodeCost() <= Options::maximumOptimizationCandidateBytecodeCost()
        && codeBlock->ownerExecutable()->isOkToOptimize();
}

bool mightInlineFunctionForCall(JITType jitType, CodeBlock* codeBlock)
{
    if (codeBlock->ownerExecutable()->inlineAttribute() != InlineAttribute::Always) {
        if (jitType == JITType::DFGJIT) {
            if (codeBlock->bytecodeCost() > Options::maximumFunctionForCallInlineCandidateBytecodeCostForDFG())
                return false;
        } else {
            if (codeBlock->bytecodeCost() > Options::maximumFunctionForCallInlineCandidateBytecodeCostForFTL())
                return false;
        }
    }
    return isSupportedForInlining(codeBlock);
}
bool mightInlineFunctionForClosureCall(JITType jitType, CodeBlock* codeBlock)
{
    if (codeBlock->ownerExecutable()->inlineAttribute() != InlineAttribute::Always) {
        if (jitType == JITType::DFGJIT) {
            if (codeBlock->bytecodeCost() > Options::maximumFunctionForClosureCallInlineCandidateBytecodeCostForDFG())
                return false;
        } else {
            if (codeBlock->bytecodeCost() > Options::maximumFunctionForClosureCallInlineCandidateBytecodeCostForFTL())
                return false;
        }
    }
    return isSupportedForInlining(codeBlock);
}
bool mightInlineFunctionForConstruct(JITType jitType, CodeBlock* codeBlock)
{
    if (codeBlock->ownerExecutable()->inlineAttribute() != InlineAttribute::Always) {
        if (jitType == JITType::DFGJIT) {
            if (codeBlock->bytecodeCost() > Options::maximumFunctionForConstructInlineCandidateBytecodeCostForDFG())
                return false;
        } else {
            if (codeBlock->bytecodeCost() > Options::maximumFunctionForConstructInlineCandidateBytecodeCostForFTL())
                return false;
        }
    }
    return isSupportedForInlining(codeBlock);
}
bool canUseOSRExitFuzzing(CodeBlock* codeBlock)
{
    return codeBlock->ownerExecutable()->canUseOSRExitFuzzing();
}

static bool verboseCapabilities()
{
    return verboseCompilationEnabled() || Options::verboseDFGFailure();
}

inline void debugFail(CodeBlock* codeBlock, OpcodeID opcodeID, CapabilityLevel result)
{
    if (verboseCapabilities() && !canCompile(result))
        dataLog("DFG rejecting opcode in ", *codeBlock, " because of opcode ", opcodeNames[opcodeID], "\n");
}

} } // namespace JSC::DFG

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif
