/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#include "FTLJITFinalizer.h"

#if ENABLE(FTL_JIT)

#include "CodeBlockWithJITType.h"
#include "DFGPlan.h"
#include "FTLState.h"
#include "ProfilerDatabase.h"
#include "ThunkGenerators.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace FTL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(JITFinalizer);

JITFinalizer::JITFinalizer(DFG::Plan& plan)
    : Finalizer(plan)
{
}

JITFinalizer::~JITFinalizer() = default;

size_t JITFinalizer::codeSize()
{
    return m_codeSize;
}

bool JITFinalizer::finalize()
{
    VM& vm = *m_plan.vm();
    WTF::crossModifyingCodeFence();

    m_plan.runMainThreadFinalizationTasks();

    CodeBlock* codeBlock = m_plan.codeBlock();
    m_jitCode->setSize(m_codeSize);
    codeBlock->setJITCode(*m_jitCode);

    if (UNLIKELY(Options::dumpFTLCodeSize())) {
        auto* baselineCodeBlock = codeBlock->baselineAlternative();
        size_t baselineCodeSize = 0;
        if (auto jitCode = baselineCodeBlock->jitCode())
            baselineCodeSize = jitCode->size();
        dataLogLn("FTL: codeSize:(", m_jitCode->size(), "),nodes:(", m_jitCode->numberOfCompiledDFGNodes(), "),baselineCodeSize:(", baselineCodeSize, "),bytecodeCost:(", baselineCodeBlock->bytecodeCost(), ")");
    }

    if (UNLIKELY(m_plan.compilation()))
        vm.m_perBytecodeProfiler->addCompilation(codeBlock, *m_plan.compilation());

    // The codeBlock is now responsible for keeping many things alive (e.g. frozen values)
    // that were previously kept alive by the plan.
    vm.writeBarrier(codeBlock);

    return true;
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
