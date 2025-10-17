/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include "FTLOSRExitHandle.h"

#if ENABLE(FTL_JIT)

#include "FTLOSRExit.h"
#include "FTLState.h"
#include "FTLThunks.h"
#include "LinkBuffer.h"
#include "ProfilerCompilation.h"

namespace JSC { namespace FTL {

void OSRExitHandle::emitExitThunk(State& state, CCallHelpers& jit)
{
    Profiler::Compilation* compilation = state.graph.compilation();
    CCallHelpers::Label myLabel = jit.label();
    label = myLabel;
    jit.pushToSaveImmediateWithoutTouchingRegisters(CCallHelpers::TrustedImm32(m_index));
    CCallHelpers::PatchableJump jump = jit.patchableJump();
    jump.linkThunk(CodeLocationLabel<JITThunkPtrTag>(state.vm().getCTIStub(osrExitGenerationThunkGenerator).code()), &jit);
    RefPtr<OSRExitHandle> self = this;
    jit.addLinkTask(
        [self, jump, myLabel, compilation] (LinkBuffer& linkBuffer) {
            self->m_jitCode->m_osrExit[self->m_index].m_patchableJump = CodeLocationJump<JSInternalPtrTag>(linkBuffer.locationOf<JSInternalPtrTag>(jump));
            if (compilation)
                compilation->addOSRExitSite({ linkBuffer.locationOf<JSInternalPtrTag>(myLabel) });
        });
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

