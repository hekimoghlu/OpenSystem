/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#include "FTLLazySlowPath.h"

#if ENABLE(FTL_JIT)

#include "LinkBuffer.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace FTL {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LazySlowPath);

LazySlowPath::~LazySlowPath() = default;

void LazySlowPath::initialize(
    CodeLocationJump<JSInternalPtrTag> patchableJump, CodeLocationLabel<JSInternalPtrTag> done,
    CodeLocationLabel<ExceptionHandlerPtrTag> exceptionTarget,
    const RegisterSetBuilder& usedRegisters, CallSiteIndex callSiteIndex, RefPtr<Generator> generator
    )
{
    m_patchableJump = patchableJump;
    m_done = done;
    m_exceptionTarget = exceptionTarget;
    m_usedRegisters = usedRegisters.buildScalarRegisterSet();
    m_callSiteIndex = callSiteIndex;
    m_generator = generator;
}

void LazySlowPath::generate(CodeBlock* codeBlock)
{
    RELEASE_ASSERT(!m_stub);

    CCallHelpers jit(codeBlock);
    GenerationParams params;
    CCallHelpers::JumpList exceptionJumps;
    params.exceptionJumps = m_exceptionTarget ? &exceptionJumps : nullptr;
    params.lazySlowPath = this;

    m_generator->run(jit, params);

    params.doneJumps.linkThunk(m_done, &jit);
    if (m_exceptionTarget)
        exceptionJumps.linkThunk(m_exceptionTarget, &jit);
    LinkBuffer linkBuffer(jit, codeBlock, LinkBuffer::Profile::FTLThunk, JITCompilationMustSucceed);
    m_stub = FINALIZE_CODE_FOR(codeBlock, linkBuffer, JITStubRoutinePtrTag, nullptr, "Lazy slow path call stub");

    MacroAssembler::repatchJump(m_patchableJump, CodeLocationLabel<JITStubRoutinePtrTag>(m_stub.code()));
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
