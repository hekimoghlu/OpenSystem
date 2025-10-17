/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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

#include "CodeBlock.h"
#include "DFGCommon.h"
#include "Options.h"

namespace JSC { namespace DFG {

#if ENABLE(DFG_JIT)
// Fast check functions; if they return true it is still necessary to
// check opcodes.
bool isSupported();
bool isSupportedForInlining(CodeBlock*);
bool mightCompileEval(CodeBlock*);
bool mightCompileProgram(CodeBlock*);
bool mightCompileFunctionForCall(CodeBlock*);
bool mightCompileFunctionForConstruct(CodeBlock*);
bool mightInlineFunctionForCall(JITType, CodeBlock*);
bool mightInlineFunctionForClosureCall(JITType, CodeBlock*);
bool mightInlineFunctionForConstruct(JITType, CodeBlock*);
bool canUseOSRExitFuzzing(CodeBlock*);
#else // ENABLE(DFG_JIT)
inline bool mightCompileEval(CodeBlock*) { return false; }
inline bool mightCompileProgram(CodeBlock*) { return false; }
inline bool mightCompileFunctionForCall(CodeBlock*) { return false; }
inline bool mightCompileFunctionForConstruct(CodeBlock*) { return false; }
inline bool mightInlineFunctionForCall(JITType, CodeBlock*) { return false; }
inline bool mightInlineFunctionForClosureCall(JITType, CodeBlock*) { return false; }
inline bool mightInlineFunctionForConstruct(JITType, CodeBlock*) { return false; }
inline bool canUseOSRExitFuzzing(CodeBlock*) { return false; }
#endif // ENABLE(DFG_JIT)

inline CapabilityLevel evalCapabilityLevel(CodeBlock* codeBlock)
{
    if (!mightCompileEval(codeBlock))
        return CannotCompile;
    
    return CanCompileAndInline;
}

inline CapabilityLevel programCapabilityLevel(CodeBlock* codeBlock)
{
    if (!mightCompileProgram(codeBlock))
        return CannotCompile;
    
    return CanCompileAndInline;
}

inline CapabilityLevel functionCapabilityLevel(bool mightCompile, bool mightInline, CapabilityLevel computedCapabilityLevel)
{
    if (mightCompile && mightInline)
        return leastUpperBound(CanCompileAndInline, computedCapabilityLevel);
    if (mightCompile && !mightInline)
        return leastUpperBound(CanCompile, computedCapabilityLevel);
    if (!mightCompile)
        return CannotCompile;
    RELEASE_ASSERT_NOT_REACHED();
    return CannotCompile;
}

inline CapabilityLevel functionForCallCapabilityLevel(JITType jitType, CodeBlock* codeBlock)
{
    return functionCapabilityLevel(
        mightCompileFunctionForCall(codeBlock),
        mightInlineFunctionForCall(jitType, codeBlock),
        CanCompileAndInline);
}

inline CapabilityLevel functionForConstructCapabilityLevel(JITType jitType, CodeBlock* codeBlock)
{
    return functionCapabilityLevel(
        mightCompileFunctionForConstruct(codeBlock),
        mightInlineFunctionForConstruct(jitType, codeBlock),
        CanCompileAndInline);
}

inline CapabilityLevel inlineFunctionForCallCapabilityLevel(JITType jitType, CodeBlock* codeBlock)
{
    if (!mightInlineFunctionForCall(jitType, codeBlock))
        return CannotCompile;
    
    return CanCompileAndInline;
}

inline CapabilityLevel inlineFunctionForClosureCallCapabilityLevel(JITType jitType, CodeBlock* codeBlock)
{
    if (!mightInlineFunctionForClosureCall(jitType, codeBlock))
        return CannotCompile;
    
    return CanCompileAndInline;
}

inline CapabilityLevel inlineFunctionForConstructCapabilityLevel(JITType jitType, CodeBlock* codeBlock)
{
    if (!mightInlineFunctionForConstruct(jitType, codeBlock))
        return CannotCompile;
    
    return CanCompileAndInline;
}

inline bool mightInlineFunctionFor(JITType jitType, CodeBlock* codeBlock, CodeSpecializationKind kind)
{
    if (kind == CodeForCall)
        return mightInlineFunctionForCall(jitType, codeBlock);
    ASSERT(kind == CodeForConstruct);
    return mightInlineFunctionForConstruct(jitType, codeBlock);
}

inline bool mightCompileFunctionFor(CodeBlock* codeBlock, CodeSpecializationKind kind)
{
    if (kind == CodeForCall)
        return mightCompileFunctionForCall(codeBlock);
    ASSERT(kind == CodeForConstruct);
    return mightCompileFunctionForConstruct(codeBlock);
}

inline bool mightInlineFunction(JITType jitType, CodeBlock* codeBlock)
{
    return mightInlineFunctionFor(jitType, codeBlock, codeBlock->specializationKind());
}

inline CapabilityLevel inlineFunctionForCapabilityLevel(JITType jitType, CodeBlock* codeBlock, CodeSpecializationKind kind, bool isClosureCall)
{
    if (isClosureCall) {
        if (kind != CodeForCall)
            return CannotCompile;
        return inlineFunctionForClosureCallCapabilityLevel(jitType, codeBlock);
    }
    if (kind == CodeForCall)
        return inlineFunctionForCallCapabilityLevel(jitType, codeBlock);
    ASSERT(kind == CodeForConstruct);
    return inlineFunctionForConstructCapabilityLevel(jitType, codeBlock);
}

inline bool isSmallEnoughToInlineCodeInto(CodeBlock* codeBlock)
{
    return codeBlock->bytecodeCost() <= Options::maximumInliningCallerBytecodeCost();
}

} } // namespace JSC::DFG
