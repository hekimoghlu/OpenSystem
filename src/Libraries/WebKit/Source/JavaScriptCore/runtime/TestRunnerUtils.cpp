/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
#include "TestRunnerUtils.h"

#include "CodeBlock.h"
#include "FunctionCodeBlock.h"
#include "JSCInlines.h"

namespace JSC {

FunctionExecutable* getExecutableForFunction(JSValue theFunctionValue)
{
    if (!theFunctionValue.isCell())
        return nullptr;

    JSFunction* theFunction = jsDynamicCast<JSFunction*>(theFunctionValue);
    if (!theFunction)
        return nullptr;
    
    FunctionExecutable* executable = jsDynamicCast<FunctionExecutable*>(
        theFunction->executable());
    return executable;
}

CodeBlock* getSomeBaselineCodeBlockForFunction(JSValue theFunctionValue)
{
    FunctionExecutable* executable = getExecutableForFunction(theFunctionValue);
    if (!executable)
        return nullptr;
    
    CodeBlock* baselineCodeBlock = executable->baselineCodeBlockFor(CodeForCall);
    
    if (!baselineCodeBlock)
        baselineCodeBlock = executable->baselineCodeBlockFor(CodeForConstruct);
    
    return baselineCodeBlock;
}

JSValue numberOfDFGCompiles(JSValue theFunctionValue)
{
    bool pretendToHaveManyCompiles = false;
#if ENABLE(DFG_JIT)
    if (!Options::useJIT() || !Options::useBaselineJIT() || !Options::useDFGJIT())
        pretendToHaveManyCompiles = true;
#else
    pretendToHaveManyCompiles = true;
#endif

    if (CodeBlock* baselineCodeBlock = getSomeBaselineCodeBlockForFunction(theFunctionValue)) {
        if (pretendToHaveManyCompiles)
            return jsNumber(1000000.0);
        return jsNumber(baselineCodeBlock->numberOfDFGCompiles());
    }
    
    return jsNumber(0);
}

JSValue setNeverInline(JSValue theFunctionValue)
{
    if (FunctionExecutable* executable = getExecutableForFunction(theFunctionValue))
        executable->setNeverInline(true);
    
    return jsUndefined();
}

JSValue setNeverOptimize(JSValue theFunctionValue)
{
    if (FunctionExecutable* executable = getExecutableForFunction(theFunctionValue))
        executable->setNeverOptimize(true);
    
    return jsUndefined();
}

JSValue optimizeNextInvocation(JSValue theFunctionValue)
{
#if ENABLE(JIT)
    if (CodeBlock* baselineCodeBlock = getSomeBaselineCodeBlockForFunction(theFunctionValue))
        baselineCodeBlock->optimizeNextInvocation();
#endif
    UNUSED_PARAM(theFunctionValue);
    return jsUndefined();
}

JSValue failNextNewCodeBlock(JSGlobalObject* globalObject)
{
    globalObject->vm().setFailNextNewCodeBlock();

    return jsUndefined();
}

JSValue numberOfDFGCompiles(JSGlobalObject*, CallFrame* callFrame)
{
    if (callFrame->argumentCount() < 1)
        return jsUndefined();
    return numberOfDFGCompiles(callFrame->uncheckedArgument(0));
}

JSValue setNeverInline(JSGlobalObject*, CallFrame* callFrame)
{
    if (callFrame->argumentCount() < 1)
        return jsUndefined();
    return setNeverInline(callFrame->uncheckedArgument(0));
}

JSValue setNeverOptimize(JSGlobalObject*, CallFrame* callFrame)
{
    if (callFrame->argumentCount() < 1)
        return jsUndefined();
    return setNeverOptimize(callFrame->uncheckedArgument(0));
}

JSValue setCannotUseOSRExitFuzzing(JSGlobalObject*, CallFrame* callFrame)
{
    if (callFrame->argumentCount() < 1)
        return jsUndefined();

    JSValue theFunctionValue = callFrame->uncheckedArgument(0);
    if (FunctionExecutable* executable = getExecutableForFunction(theFunctionValue))
        executable->setCanUseOSRExitFuzzing(false);

    return jsUndefined();
}

JSValue optimizeNextInvocation(JSGlobalObject*, CallFrame* callFrame)
{
    if (callFrame->argumentCount() < 1)
        return jsUndefined();
    return optimizeNextInvocation(callFrame->uncheckedArgument(0));
}

// This is a hook called at the bitter end of some of our tests.
void finalizeStatsAtEndOfTesting()
{
}

} // namespace JSC

