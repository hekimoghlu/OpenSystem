/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include "FTLOSRExit.h"

#if ENABLE(FTL_JIT)

#include "B3StackmapGenerationParams.h"
#include "FTLJITCode.h"
#include "FTLState.h"

namespace JSC { namespace FTL {

using namespace B3;
using namespace DFG;

OSRExitDescriptor::OSRExitDescriptor(
    DataFormat profileDataFormat, MethodOfGettingAValueProfile valueProfile,
    unsigned numberOfArguments, unsigned numberOfLocals, unsigned numberOfTmps)
    : m_profileDataFormat(profileDataFormat)
    , m_valueProfile(valueProfile)
    , m_values(numberOfArguments, numberOfLocals, numberOfTmps)
{
}

void OSRExitDescriptor::validateReferences(const TrackedReferences& trackedReferences)
{
    for (unsigned i = m_values.size(); i--;)
        m_values[i].validateReferences(trackedReferences);
    
    for (ExitTimeObjectMaterialization* materialization : m_materializations)
        materialization->validateReferences(trackedReferences);
}

Ref<OSRExitHandle> OSRExitDescriptor::emitOSRExit(
    State& state, ExitKind exitKind, const NodeOrigin& nodeOrigin, CCallHelpers& jit,
    const StackmapGenerationParams& params, uint32_t dfgNodeIndex, unsigned offset)
{
    Ref<OSRExitHandle> handle =
        prepareOSRExitHandle(state, exitKind, nodeOrigin, params, dfgNodeIndex, offset);
    handle->emitExitThunk(state, jit);
    return handle;
}

Ref<OSRExitHandle> OSRExitDescriptor::emitOSRExitLater(
    State& state, ExitKind exitKind, const NodeOrigin& nodeOrigin,
    const StackmapGenerationParams& params, uint32_t dfgNodeIndex, unsigned offset)
{
    RefPtr<OSRExitHandle> handle =
        prepareOSRExitHandle(state, exitKind, nodeOrigin, params, dfgNodeIndex, offset);
    params.addLatePath(
        [handle, &state] (CCallHelpers& jit) {
            handle->emitExitThunk(state, jit);
        });
    return handle.releaseNonNull();
}

Ref<OSRExitHandle> OSRExitDescriptor::prepareOSRExitHandle(
    State& state, ExitKind exitKind, const NodeOrigin& nodeOrigin,
    const StackmapGenerationParams& params, uint32_t dfgNodeIndex, unsigned offset)
{
    FixedVector<B3::ValueRep> valueReps(params.size() - offset);
    for (unsigned i = offset, indexInValueReps = 0; i < params.size(); ++i, ++indexInValueReps)
        valueReps[indexInValueReps] = params[i];
    unsigned index = state.jitCode->m_osrExit.size();
    state.jitCode->m_osrExit.append(OSRExit(this, exitKind, nodeOrigin.forExit, nodeOrigin.semantic, nodeOrigin.wasHoisted, dfgNodeIndex, WTFMove(valueReps)));
    return adoptRef(*new OSRExitHandle(index, state.jitCode.get()));
}

OSRExit::OSRExit(
    OSRExitDescriptor* descriptor, ExitKind exitKind, CodeOrigin codeOrigin,
    CodeOrigin codeOriginForExitProfile, bool wasHoisted, uint32_t dfgNodeIndex, FixedVector<B3::ValueRep>&& valueReps)
    : OSRExitBase(exitKind, codeOrigin, codeOriginForExitProfile, wasHoisted, dfgNodeIndex)
    , m_descriptor(descriptor)
    , m_valueReps(WTFMove(valueReps))
{
}

CodeLocationJump<JSInternalPtrTag> OSRExit::codeLocationForRepatch(CodeBlock* ftlCodeBlock) const
{
    UNUSED_PARAM(ftlCodeBlock);
    return m_patchableJump;
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
