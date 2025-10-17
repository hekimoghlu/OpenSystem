/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#include "DFGJITFinalizer.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "CodeBlockWithJITType.h"
#include "DFGPlan.h"
#include "HeapInlines.h"
#include "ProfilerDatabase.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace DFG {

WTF_MAKE_TZONE_ALLOCATED_IMPL(JITFinalizer);

JITFinalizer::JITFinalizer(Plan& plan, Ref<DFG::JITCode>&& jitCode, CodePtr<JSEntryPtrTag> withArityCheck)
    : Finalizer(plan)
    , m_jitCode(WTFMove(jitCode))
    , m_withArityCheck(withArityCheck)
{
}

JITFinalizer::~JITFinalizer() = default;

size_t JITFinalizer::codeSize()
{
    return m_jitCode->size();
}

bool JITFinalizer::finalize()
{
    VM& vm = *m_plan.vm();

    WTF::crossModifyingCodeFence();

    m_plan.runMainThreadFinalizationTasks();

    CodeBlock* codeBlock = m_plan.codeBlock();

    codeBlock->setJITCode(m_jitCode.copyRef());

    auto data = m_plan.tryFinalizeJITData(m_jitCode.get());
    if (UNLIKELY(!data))
        return false;
    codeBlock->setDFGJITData(WTFMove(data));

#if ENABLE(FTL_JIT)
    m_jitCode->optimizeAfterWarmUp(codeBlock);
#endif // ENABLE(FTL_JIT)

    if (UNLIKELY(m_plan.compilation()))
        vm.m_perBytecodeProfiler->addCompilation(codeBlock, *m_plan.compilation());

    if (!m_plan.willTryToTierUp())
        codeBlock->baselineVersion()->m_didFailFTLCompilation = true;

    // The codeBlock is now responsible for keeping many things alive (e.g. frozen values)
    // that were previously kept alive by the plan.
    vm.writeBarrier(codeBlock);

    return true;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
