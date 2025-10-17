/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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
#include "DFGDriver.h"

#include "CodeBlock.h"
#include "DFGJITCode.h"
#include "DFGPlan.h"
#include "DFGThunks.h"
#include "FunctionAllowlist.h"
#include "JITCode.h"
#include "JITWorklist.h"
#include "Options.h"
#include "ThunkGenerators.h"
#include "TypeProfilerLog.h"
#include <wtf/NeverDestroyed.h>

namespace JSC { namespace DFG {

static unsigned numCompilations;

unsigned getNumCompilations()
{
    return numCompilations;
}

#if ENABLE(DFG_JIT)
static FunctionAllowlist& ensureGlobalDFGAllowlist()
{
    static LazyNeverDestroyed<FunctionAllowlist> dfgAllowlist;
    static std::once_flag initializeAllowlistFlag;
    std::call_once(initializeAllowlistFlag, [] {
        const char* functionAllowlistFile = Options::dfgAllowlist();
        dfgAllowlist.construct(functionAllowlistFile);
    });
    return dfgAllowlist;
}

static CompilationResult compileImpl(
    VM& vm, CodeBlock* codeBlock, CodeBlock* profiledDFGCodeBlock, JITCompilationMode mode,
    BytecodeIndex osrEntryBytecodeIndex, Operands<std::optional<JSValue>>&& mustHandleValues,
    Ref<DeferredCompilationCallback>&& callback)
{
    if (!Options::bytecodeRangeToDFGCompile().isInRange(codeBlock->instructionsSize())
        || !ensureGlobalDFGAllowlist().contains(codeBlock))
        return CompilationFailed;
    
    numCompilations++;
    
    ASSERT(codeBlock);
    ASSERT(codeBlock->alternative());
    ASSERT(JITCode::isBaselineCode(codeBlock->alternative()->jitType()));
    ASSERT(!profiledDFGCodeBlock || profiledDFGCodeBlock->jitType() == JITType::DFGJIT);
    
    if (logCompilationChanges(mode))
        dataLog("DFG(Driver) compiling ", *codeBlock, " with ", mode, ", instructions size = ", codeBlock->instructionsSize(), "\n");
    
    if (vm.typeProfiler())
        vm.typeProfilerLog()->processLogEntries(vm, "Preparing for DFG compilation."_s);
    
    Ref<Plan> plan = adoptRef(*new Plan(codeBlock, profiledDFGCodeBlock, mode, osrEntryBytecodeIndex, WTFMove(mustHandleValues)));

    plan->setCallback(WTFMove(callback));
    JITWorklist& worklist = JITWorklist::ensureGlobalWorklist();
    dataLogLnIf(Options::useConcurrentJIT() && logCompilationChanges(mode), "Deferring DFG compilation of ", *codeBlock, " with queue length ", worklist.queueLength(), ".\n");
    return worklist.enqueue(WTFMove(plan));
}
#else // ENABLE(DFG_JIT)
static CompilationResult compileImpl(
    VM&, CodeBlock*, CodeBlock*, JITCompilationMode, BytecodeIndex, const Operands<std::optional<JSValue>>&,
    Ref<DeferredCompilationCallback>&&)
{
    return CompilationFailed;
}
#endif // ENABLE(DFG_JIT)

CompilationResult compile(
    VM& vm, CodeBlock* codeBlock, CodeBlock* profiledDFGCodeBlock, JITCompilationMode mode,
    BytecodeIndex osrEntryBytecodeIndex, Operands<std::optional<JSValue>>&& mustHandleValues,
    Ref<DeferredCompilationCallback>&& callback)
{
    return compileImpl(vm, codeBlock, profiledDFGCodeBlock, mode, osrEntryBytecodeIndex, WTFMove(mustHandleValues), callback.copyRef());
}

} } // namespace JSC::DFG
