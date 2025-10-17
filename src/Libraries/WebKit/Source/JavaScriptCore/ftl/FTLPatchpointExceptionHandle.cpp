/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include "FTLPatchpointExceptionHandle.h"

#if ENABLE(FTL_JIT)

#include "B3StackmapGenerationParams.h"
#include "FTLExceptionTarget.h"
#include "FTLOSRExit.h"
#include "FTLOSRExitHandle.h"
#include "FTLState.h"

namespace JSC { namespace FTL {

using namespace DFG;

Ref<PatchpointExceptionHandle> PatchpointExceptionHandle::create(
    State& state, OSRExitDescriptor* descriptor, NodeOrigin origin, unsigned dfgNodeIndex, unsigned offset,
    const HandlerInfo& handler)
{
    return adoptRef(*new PatchpointExceptionHandle(state, descriptor, origin, dfgNodeIndex, offset, handler));
}

RefPtr<PatchpointExceptionHandle> PatchpointExceptionHandle::defaultHandle(State& state, unsigned dfgNodeIndex)
{
    if (!state.defaultExceptionHandle) {
        state.defaultExceptionHandle = adoptRef(
            new PatchpointExceptionHandle(state, nullptr, NodeOrigin(), dfgNodeIndex, 0, HandlerInfo()));
    }
    return state.defaultExceptionHandle;
}

PatchpointExceptionHandle::~PatchpointExceptionHandle() = default;

RefPtr<ExceptionTarget> PatchpointExceptionHandle::scheduleExitCreation(
    const B3::StackmapGenerationParams& params)
{
    if (!m_descriptor) {
        // NOTE: This object could be a singleton, however usually we toss the ExceptionHandler
        // object shortly after creation.
        bool isDefaultHandler = true;
        return adoptRef(
            new ExceptionTarget(isDefaultHandler, m_state.exceptionHandler, nullptr));
    }
    bool isDefaultHandler = false;
    return adoptRef(new ExceptionTarget(isDefaultHandler, { }, createHandle(ExceptionCheck, params)));
}

void PatchpointExceptionHandle::scheduleExitCreationForUnwind(
    const B3::StackmapGenerationParams& params, CallSiteIndex callSiteIndex)
{
    if (!m_descriptor)
        return;
    
    RefPtr<OSRExitHandle> handle = createHandle(GenericUnwind, params);

    handle->m_jitCode->m_osrExit[handle->m_index].m_exceptionHandlerCallSiteIndex = callSiteIndex;

    HandlerInfo handler = m_handler;
    auto* plan = &m_state.graph.m_plan;
    params.addLatePath(
        [handle, handler, callSiteIndex, plan] (CCallHelpers& jit) {
            CodeBlock* codeBlock = jit.codeBlock();
            jit.addLinkTask([=] (LinkBuffer& linkBuffer) {
                HandlerInfo newHandler = handler;
                newHandler.start = callSiteIndex.bits();
                newHandler.end = callSiteIndex.bits() + 1;
                newHandler.nativeCode = linkBuffer.locationOf<ExceptionHandlerPtrTag>(handle->label);
                plan->addMainThreadFinalizationTask([=] {
                    codeBlock->appendExceptionHandler(newHandler);
                });
            });
        });
}

PatchpointExceptionHandle::PatchpointExceptionHandle(
    State& state, OSRExitDescriptor* descriptor, NodeOrigin origin, unsigned dfgNodeIndex, unsigned offset,
    const HandlerInfo& handler)
    : m_state(state)
    , m_descriptor(descriptor)
    , m_origin(origin)
    , m_dfgNodeIndex(dfgNodeIndex)
    , m_offset(offset)
    , m_handler(handler)
{
}

Ref<OSRExitHandle> PatchpointExceptionHandle::createHandle(
    ExitKind kind, const B3::StackmapGenerationParams& params)
{
    return m_descriptor->emitOSRExitLater(
        m_state, kind, m_origin, params, m_dfgNodeIndex, m_offset);
}

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)

