/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include "WasmBinding.h"

#include "CallFrame.h"

#if ENABLE(WEBASSEMBLY) && ENABLE(JIT)

#include "CCallHelpers.h"
#include "DisallowMacroScratchRegisterUsage.h"
#include "JSWebAssemblyInstance.h"
#include "LinkBuffer.h"
#include "WasmCallingConvention.h"

namespace JSC { namespace Wasm {

using JIT = CCallHelpers;

Expected<MacroAssemblerCodeRef<WasmEntryPtrTag>, BindingFailure> wasmToWasm(unsigned importIndex)
{
    // FIXME: Consider uniquify the stubs based on signature + index to see if this saves memory.
    // https://bugs.webkit.org/show_bug.cgi?id=184157
    JIT jit;

    GPRReg scratch = wasmCallingConvention().prologueScratchGPRs[0];
    ASSERT(scratch != GPRReg::InvalidGPRReg);
    ASSERT(noOverlap(scratch, GPRInfo::wasmContextInstancePointer));

    JIT_COMMENT(jit, "Store Callee's wasm callee for import function ", importIndex);
    jit.loadPtr(JIT::Address(GPRInfo::wasmContextInstancePointer, JSWebAssemblyInstance::offsetOfBoxedWasmCalleeLoadLocation(importIndex)), scratch);
    jit.loadPtr(JIT::Address(scratch), scratch);
    // We are halfway between being the caller and the callee: we have already made the call, but not yet completed the prologue.
    // On ARM64 this doesn't really matter, but on intel we need to worry about the pushed pc.
    jit.storeWasmCalleeCallee(scratch, safeCast<int>(sizeof(CallerFrameAndPC)) - safeCast<int>(prologueStackPointerDelta()));

    // FIXME: This could be a load pair.
    // B3's call codegen ensures that the JSCell is a WebAssemblyFunction.
    // While we're accessing that cacheline, also get the wasm entrypoint so we can tail call to it below.

    jit.loadPtr(JIT::Address(GPRInfo::wasmContextInstancePointer, JSWebAssemblyInstance::offsetOfEntrypointLoadLocation(importIndex)), scratch);
    // Get the callee's JSWebAssemblyInstance and set it as WasmContext's instance. The caller will take care of restoring its own JSWebAssemblyInstance.
    // This switches the current instance.
    jit.loadPtr(JIT::Address(GPRInfo::wasmContextInstancePointer, JSWebAssemblyInstance::offsetOfTargetInstance(importIndex)), GPRInfo::wasmContextInstancePointer); // JSWebAssemblyInstance*.

#if !CPU(ARM) // ARM has no pinned registers for Wasm Memory, so no need to set them up
    // FIXME the following code assumes that all JSWebAssemblyInstance have the same pinned registers. https://bugs.webkit.org/show_bug.cgi?id=162952
    // Set up the callee's baseMemoryPointer register as well as the memory size registers.
    {
        jit.loadPairPtr(GPRInfo::wasmContextInstancePointer, CCallHelpers::TrustedImm32(JSWebAssemblyInstance::offsetOfCachedMemory()), GPRInfo::wasmBaseMemoryPointer, GPRInfo::wasmBoundsCheckingSizeRegister);
        jit.cageConditionally(Gigacage::Primitive, GPRInfo::wasmBaseMemoryPointer, GPRInfo::wasmBoundsCheckingSizeRegister, wasmCallingConvention().prologueScratchGPRs[1]);
    }
#endif

    // Tail call into the callee WebAssembly function.
    jit.loadPtr(JIT::Address(scratch), scratch);
    jit.farJump(scratch, WasmEntryPtrTag);

    LinkBuffer patchBuffer(jit, GLOBAL_THUNK_ID, LinkBuffer::Profile::WasmThunk, JITCompilationCanFail);
    if (UNLIKELY(patchBuffer.didFailToAllocate()))
        return makeUnexpected(BindingFailure::OutOfMemory);

    return FINALIZE_WASM_CODE(patchBuffer, WasmEntryPtrTag, nullptr, "WebAssembly->WebAssembly import[%i]", importIndex);
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY) && ENABLE(JIT)
