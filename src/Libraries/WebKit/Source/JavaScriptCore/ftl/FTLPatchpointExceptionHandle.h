/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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

#include "DFGCommon.h"

#if ENABLE(FTL_JIT)

#include "CallFrame.h"
#include "DFGNodeOrigin.h"
#include "ExitKind.h"
#include "HandlerInfo.h"
#include <wtf/Ref.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC {

namespace B3 {
class StackmapGenerationParams;
} // namespace B3

namespace FTL {

class ExceptionTarget;
class State;
struct OSRExitDescriptor;
struct OSRExitHandle;

class PatchpointExceptionHandle : public ThreadSafeRefCounted<PatchpointExceptionHandle> {
public:
    static Ref<PatchpointExceptionHandle> create(
        State&, OSRExitDescriptor*, DFG::NodeOrigin, unsigned dfgNodeIndex, unsigned offset, const HandlerInfo&);

    static RefPtr<PatchpointExceptionHandle> defaultHandle(State&, unsigned dfgNodeIndex);
    
    ~PatchpointExceptionHandle();

    // Note that you can use this handle to schedule any number of exits. This capability is here for
    // two reasons:
    // 
    // - B3 code duplication. B3 could take a patchpoint and turn it into multiple patchpoints if it
    //   duplicates code. Duplicating code is legal since you can do it without changing the behavior
    //   of the program. One example is tail duplication. Another is jump threading. Yet another is
    //   path specialization. You will have one PatchpointExceptionHandle per patchpoint you create
    //   during DFG->B3 lowering, and that patchpoint will have a generator that calls
    //   handle->scheduleBlah(). That generator will be called multiple times if your patchpoint got
    //   duplicated.
    //
    // - Combination of unwind and non-unwind exception handlers inside one patchpoint. A GetById may
    //   need both an exception handler that serves as an unwind target and an exception handler that
    //   is branched to directly for operation calls emitted inside the patchpoint. In that case,
    //   you'll call both scheduleExitCreation() and scheduleExitCreationForUnwind() on the same
    //   handle.

    // Schedules the creation of an OSR exit jump destination. You don't know when this will be
    // created, but it will happen before linking. You can link jumps to it during link time. That's
    // why this returns an ExceptionTarget. That will contain the jump destination (target->label())
    // at link time. This function should be used for exceptions from C calls.
    RefPtr<ExceptionTarget> scheduleExitCreation(const B3::StackmapGenerationParams&);

    // Schedules the creation of an OSR exit jump destination, and ensures that it gets associated
    // with the handler for some callsite index. This function should be used for exceptions from JS.
    void scheduleExitCreationForUnwind(const B3::StackmapGenerationParams&, CallSiteIndex);

private:
    PatchpointExceptionHandle(
        State&, OSRExitDescriptor*, DFG::NodeOrigin, unsigned dfgNodeIndex, unsigned offset, const HandlerInfo&);

    Ref<OSRExitHandle> createHandle(ExitKind, const B3::StackmapGenerationParams&);

    State& m_state;
    OSRExitDescriptor* m_descriptor;
    DFG::NodeOrigin m_origin;
    unsigned m_dfgNodeIndex;
    unsigned m_offset;
    HandlerInfo m_handler;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
