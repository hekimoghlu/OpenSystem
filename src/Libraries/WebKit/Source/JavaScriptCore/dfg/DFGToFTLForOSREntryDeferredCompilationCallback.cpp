/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#include "DFGToFTLForOSREntryDeferredCompilationCallback.h"

#if ENABLE(FTL_JIT)

#include "CodeBlock.h"
#include "DFGJITCode.h"
#include "FTLForOSREntryJITCode.h"

namespace JSC { namespace DFG {

ToFTLForOSREntryDeferredCompilationCallback::ToFTLForOSREntryDeferredCompilationCallback(JITCode::TriggerReason* forcedOSREntryTrigger)
    : m_forcedOSREntryTrigger(forcedOSREntryTrigger)
{
}

ToFTLForOSREntryDeferredCompilationCallback::~ToFTLForOSREntryDeferredCompilationCallback() = default;

Ref<ToFTLForOSREntryDeferredCompilationCallback>ToFTLForOSREntryDeferredCompilationCallback::create(JITCode::TriggerReason* forcedOSREntryTrigger)
{
    return adoptRef(*new ToFTLForOSREntryDeferredCompilationCallback(forcedOSREntryTrigger));
}

void ToFTLForOSREntryDeferredCompilationCallback::compilationDidBecomeReadyAsynchronously(
    CodeBlock* codeBlock, CodeBlock* profiledDFGCodeBlock)
{
    dataLogLnIf(Options::verboseOSR(),
        "Optimizing compilation of ", *codeBlock, " (for ", *profiledDFGCodeBlock,
        ") did become ready.");

    *m_forcedOSREntryTrigger = JITCode::TriggerReason::CompilationDone;
}

void ToFTLForOSREntryDeferredCompilationCallback::compilationDidComplete(
    CodeBlock* codeBlock, CodeBlock* profiledDFGCodeBlock, CompilationResult result)
{
    dataLogLnIf(Options::verboseOSR(),
        "Optimizing compilation of ", *codeBlock, " (for ", *profiledDFGCodeBlock,
        ") result: ", result);
    
    JITCode* jitCode = profiledDFGCodeBlock->jitCode()->dfg();
        
    switch (result) {
    case CompilationSuccessful: {
        jitCode->setOSREntryBlock(codeBlock->vm(), profiledDFGCodeBlock, codeBlock);
        BytecodeIndex osrEntryBytecode = codeBlock->jitCode()->ftlForOSREntry()->bytecodeIndex();
        jitCode->tierUpEntryTriggers.set(osrEntryBytecode, JITCode::TriggerReason::CompilationDone);
        break;
    }
    case CompilationFailed:
        jitCode->osrEntryRetry = 0;
        jitCode->abandonOSREntry = true;
        profiledDFGCodeBlock->jitCode()->dfg()->setOptimizationThresholdBasedOnCompilationResult(
            profiledDFGCodeBlock, result);
        break;
    case CompilationDeferred:
        RELEASE_ASSERT_NOT_REACHED();
    case CompilationInvalidated:
        jitCode->osrEntryRetry = 0;
        break;
    }
    
    DeferredCompilationCallback::compilationDidComplete(codeBlock, profiledDFGCodeBlock, result);
}

} } // JSC::DFG

#endif // ENABLE(FTL_JIT)
