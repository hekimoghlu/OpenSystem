/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#include "DFGToFTLDeferredCompilationCallback.h"

#if ENABLE(FTL_JIT)

#include "CodeBlock.h"
#include "DFGJITCode.h"

namespace JSC { namespace DFG {

ToFTLDeferredCompilationCallback::ToFTLDeferredCompilationCallback() = default;

ToFTLDeferredCompilationCallback::~ToFTLDeferredCompilationCallback() = default;

Ref<ToFTLDeferredCompilationCallback> ToFTLDeferredCompilationCallback::create()
{
    return adoptRef(*new ToFTLDeferredCompilationCallback());
}

void ToFTLDeferredCompilationCallback::compilationDidBecomeReadyAsynchronously(
    CodeBlock* codeBlock, CodeBlock* profiledDFGCodeBlock)
{
    dataLogLnIf(Options::verboseOSR(),
        "Optimizing compilation of ", codeBlock, " (for ", profiledDFGCodeBlock,
        ") did become ready.");
    
    profiledDFGCodeBlock->jitCode()->dfg()->forceOptimizationSlowPathConcurrently(
        profiledDFGCodeBlock);
}

void ToFTLDeferredCompilationCallback::compilationDidComplete(
    CodeBlock* codeBlock, CodeBlock* profiledDFGCodeBlock, CompilationResult result)
{
    dataLogLnIf(Options::verboseOSR(),
        "Optimizing compilation of ", codeBlock, " (for ", profiledDFGCodeBlock,
        ") result: ", result);
    
    if (profiledDFGCodeBlock->replacement() != profiledDFGCodeBlock) {
        dataLogLnIf(Options::verboseOSR(),
            "Dropping FTL code block ", codeBlock, " on the floor because the "
            "DFG code block ", profiledDFGCodeBlock, " was jettisoned.");
        return;
    }
    
    if (result == CompilationSuccessful)
        codeBlock->ownerExecutable()->installCode(codeBlock);
    
    profiledDFGCodeBlock->jitCode()->dfg()->setOptimizationThresholdBasedOnCompilationResult(
        profiledDFGCodeBlock, result);

    DeferredCompilationCallback::compilationDidComplete(codeBlock, profiledDFGCodeBlock, result);
}

} } // JSC::DFG

#endif // ENABLE(FTL_JIT)
