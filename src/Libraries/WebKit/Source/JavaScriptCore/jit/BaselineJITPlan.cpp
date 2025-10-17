/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include "BaselineJITPlan.h"

#include "JITSafepoint.h"

#if ENABLE(JIT)

namespace JSC {

BaselineJITPlan::BaselineJITPlan(CodeBlock* codeBlock)
    : JITPlan(JITCompilationMode::Baseline, codeBlock)
{
    JIT::doMainThreadPreparationBeforeCompile(codeBlock->vm());
}

auto BaselineJITPlan::compileInThreadImpl(JITCompilationEffort effort) -> CompilationPath
{
    // BaselineJITPlan can keep underlying CodeBlock alive while running.
    // So we do not need to suspend this compilation thread while running GC.
    Safepoint::Result result;
    {
        Safepoint safepoint(*this, result);
        safepoint.begin(false);

        JIT jit(*m_vm, *this, m_codeBlock);
        auto jitCode = jit.compileAndLinkWithoutFinalizing(effort);
        m_jitCode = WTFMove(jitCode);
    }
    if (result.didGetCancelled())
        return CancelPath;
    return BaselinePath;
}

auto BaselineJITPlan::compileInThreadImpl() -> CompilationPath
{
    return compileInThreadImpl(JITCompilationCanFail);
}

auto BaselineJITPlan::compileSync(JITCompilationEffort effort) -> CompilationPath
{
    return compileInThreadImpl(effort);
}

size_t BaselineJITPlan::codeSize() const
{
    if (m_jitCode)
        return m_jitCode->size();
    return 0;
}

bool BaselineJITPlan::isKnownToBeLiveAfterGC()
{
    // If stage is not JITPlanStage::Canceled, we should keep this alive and mark underlying CodeBlock anyway.
    // Regardless of whether the owner ScriptExecutable / CodeBlock dies, compiled code would be still usable
    // since Baseline JIT is *unlinked*. So, let's not stop compilation.
    return m_stage != JITPlanStage::Canceled;
}

bool BaselineJITPlan::isKnownToBeLiveDuringGC(AbstractSlotVisitor&)
{
    // Ditto to isKnownToBeLiveAfterGC. Unless plan gets completely cancelled before running, we should keep compilation running.
    return m_stage != JITPlanStage::Canceled;
}

CompilationResult BaselineJITPlan::finalize()
{
    CompilationResult result = JIT::finalizeOnMainThread(m_codeBlock, *this, m_jitCode);
    switch (result) {
    case CompilationFailed:
        CODEBLOCK_LOG_EVENT(m_codeBlock, "delayJITCompile", ("compilation failed"));
        dataLogLnIf(Options::verboseOSR(), "    JIT compilation failed.");
        m_codeBlock->dontJITAnytimeSoon();
        m_codeBlock->m_didFailJITCompilation = true;
        break;
    case CompilationSuccessful:
        WTF::crossModifyingCodeFence();
        dataLogLnIf(Options::verboseOSR(), "    JIT compilation successful.");
        m_codeBlock->ownerExecutable()->installCode(m_codeBlock);
        m_codeBlock->jitSoon();
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }

    return result;
}

} // namespace JSC

#endif // ENABLE(JIT)
