/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#include "ProbeContext.h"

#if ENABLE(ASSEMBLER)

#include <wtf/TZoneMallocInlines.h>

namespace JSC {
namespace Probe {

static void SYSV_ABI flushDirtyStackPages(State*);

extern "C" void SYSV_ABI executeJSCJITProbe(State* state)
{
    Context context(state);
#if CPU(ARM64)
    auto& cpu = context.cpu;
    void* originalLR = cpu.gpr<void*>(ARM64Registers::lr);
    void* originalPC = cpu.pc();
#endif

    state->initializeStackFunction = nullptr;
    state->initializeStackArg = nullptr;
    state->probeFunction(context);

#if CPU(ARM64)
    // The ARM64 probe trampoline does not support changing both lr and pc.
    RELEASE_ASSERT(originalPC == cpu.pc() || originalLR == cpu.gpr<void*>(ARM64Registers::lr));
#endif

    if (context.hasWritesToFlush()) {
        context.stack().setSavedStackPointer(state->cpu.sp());
        void* lowWatermark = context.stack().lowWatermarkFromVisitingDirtyPages();
        state->cpu.sp() = std::min(lowWatermark, state->cpu.sp());

        state->initializeStackFunction = flushDirtyStackPages;
        state->initializeStackArg = context.releaseStack();
    }
}

static void SYSV_ABI flushDirtyStackPages(State* state)
{
    std::unique_ptr<Stack> stack(reinterpret_cast<Probe::Stack*>(state->initializeStackArg));
    stack->flushWrites();
    state->cpu.sp() = stack->savedStackPointer();
}

// Not for general use. This should only be for writing tests.
JS_EXPORT_PRIVATE void* probeStateForContext(Context&);
void* probeStateForContext(Context& context)
{
    return context.m_state;
}

} // namespace Probe
} // namespace JSC

#endif // ENABLE(ASSEMBLER)
