/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#include "DFGThunks.h"

#if ENABLE(DFG_JIT)

#include "AssemblyHelpersSpoolers.h"
#include "CCallHelpers.h"
#include "DFGJITCode.h"
#include "DFGOSRExit.h"
#include "DFGOSRExitCompilerCommon.h"
#include "FPRInfo.h"
#include "GPRInfo.h"
#include "LinkBuffer.h"
#include "MacroAssembler.h"
#include "ProbeContext.h"

namespace JSC { namespace DFG {

MacroAssemblerCodeRef<JITThunkPtrTag> osrExitGenerationThunkGenerator(VM& vm)
{
    CCallHelpers jit(nullptr);

    // This needs to happen before we use the scratch buffer because this function also uses the scratch buffer.
    adjustFrameAndStackInOSRExitCompilerThunk<DFG::JITCode>(jit, vm, JITType::DFGJIT);

#if USE(JSVALUE64)
    jit.store32(GPRInfo::numberTagRegister, &vm.osrExitIndex);
#endif

    size_t scratchSize = sizeof(EncodedJSValue) * (GPRInfo::numberOfRegisters + FPRInfo::numberOfRegisters);
    ScratchBuffer* scratchBuffer = vm.scratchBufferForSize(scratchSize);
    EncodedJSValue* buffer = static_cast<EncodedJSValue*>(scratchBuffer->dataBuffer());

#if CPU(ARM64)
    constexpr GPRReg bufferGPR = CCallHelpers::memoryTempRegister;
    constexpr unsigned firstGPR = 0;
#elif CPU(X86_64)
    GPRReg bufferGPR = jit.scratchRegister();
    constexpr unsigned firstGPR = 0;
#else
    GPRReg bufferGPR = GPRInfo::toRegister(0);
    constexpr unsigned firstGPR = 1;
#endif

    if constexpr (firstGPR) {
        // We're using the firstGPR as the bufferGPR, and need to save it manually.
        RELEASE_ASSERT(GPRInfo::numberOfRegisters >= 1);
        RELEASE_ASSERT(bufferGPR == GPRInfo::toRegister(0));
#if USE(JSVALUE64)
        jit.store64(bufferGPR, buffer);
#else
        jit.store32(bufferGPR, buffer);
#endif
    }

    jit.move(CCallHelpers::TrustedImmPtr(buffer), bufferGPR);

    CCallHelpers::StoreRegSpooler storeSpooler(jit, bufferGPR);

    for (unsigned i = firstGPR; i < GPRInfo::numberOfRegisters; ++i) {
        ptrdiff_t offset = i * sizeof(CPURegister);
        storeSpooler.storeGPR({ GPRInfo::toRegister(i), offset, conservativeWidthWithoutVectors(GPRInfo::toRegister(i)) });
    }
    storeSpooler.finalizeGPR();

    for (unsigned i = 0; i < FPRInfo::numberOfRegisters; ++i) {
        ptrdiff_t offset = (GPRInfo::numberOfRegisters + i) * sizeof(double);
        storeSpooler.storeFPR({ FPRInfo::toRegister(i), offset, conservativeWidthWithoutVectors(FPRInfo::toRegister(i)) });
    }
    storeSpooler.finalizeFPR();

    // This will implicitly pass GPRInfo::callFrameRegister as the first argument based on the operation type.
    jit.setupArguments<decltype(operationCompileOSRExit)>(bufferGPR);
    jit.prepareCallOperation(vm);
    jit.callOperation<OperationPtrTag>(operationCompileOSRExit);

    jit.move(CCallHelpers::TrustedImmPtr(buffer), bufferGPR);
    CCallHelpers::LoadRegSpooler loadSpooler(jit, bufferGPR);

    for (unsigned i = firstGPR; i < GPRInfo::numberOfRegisters; ++i) {
        ptrdiff_t offset = i * sizeof(CPURegister);
        loadSpooler.loadGPR({ GPRInfo::toRegister(i), offset, conservativeWidthWithoutVectors(GPRInfo::toRegister(i)) });
    }
    loadSpooler.finalizeGPR();

    for (unsigned i = 0; i < FPRInfo::numberOfRegisters; ++i) {
        ptrdiff_t offset = (GPRInfo::numberOfRegisters + i) * sizeof(double);
        loadSpooler.loadFPR({ FPRInfo::toRegister(i), offset, conservativeWidthWithoutVectors(FPRInfo::toRegister(i)) });
    }
    loadSpooler.finalizeFPR();

    if constexpr (firstGPR) {
        // We're using the firstGPR as the bufferGPR, and need to restore it manually.
        ASSERT(bufferGPR == GPRInfo::toRegister(0));
#if USE(JSVALUE64)
        jit.load64(buffer, bufferGPR);
#else
        jit.load32(buffer, bufferGPR);
#endif
    }

    jit.farJump(MacroAssembler::AbsoluteAddress(&vm.osrExitJumpDestination), OSRExitPtrTag);

    LinkBuffer patchBuffer(jit, GLOBAL_THUNK_ID, LinkBuffer::Profile::DFGThunk);
    return FINALIZE_THUNK(patchBuffer, JITThunkPtrTag, nullptr, "DFG OSR exit generation thunk");
}

MacroAssemblerCodeRef<JITThunkPtrTag> osrEntryThunkGenerator(VM& vm)
{
    AssemblyHelpers jit(nullptr);

    // We get passed the address of a scratch buffer in GPRInfo::returnValueGPR2.
    // The first 8-byte slot of the buffer is the frame size. The second 8-byte slot
    // is the pointer to where we are supposed to jump. The remaining bytes are
    // the new call frame header followed by the locals.
    
    ptrdiff_t offsetOfFrameSize = 0; // This is the DFG frame count.
    ptrdiff_t offsetOfTargetPC = offsetOfFrameSize + sizeof(EncodedJSValue);
    ptrdiff_t offsetOfPayload = offsetOfTargetPC + sizeof(EncodedJSValue);
    ptrdiff_t offsetOfLocals = offsetOfPayload + sizeof(Register) * CallFrame::headerSizeInRegisters;
    
    jit.move(GPRInfo::returnValueGPR2, GPRInfo::regT0);
    jit.loadPtr(MacroAssembler::Address(GPRInfo::regT0, offsetOfFrameSize), GPRInfo::regT1); // Load the frame size.
    jit.negPtr(GPRInfo::regT1, GPRInfo::regT2);
    jit.getEffectiveAddress(MacroAssembler::BaseIndex(GPRInfo::callFrameRegister, GPRInfo::regT2, MacroAssembler::TimesEight), MacroAssembler::stackPointerRegister);
    
    // Copying locals and header from scratch buffer to the new CallFrame. This also replaces
    MacroAssembler::Label loop = jit.label();
    jit.subPtr(MacroAssembler::TrustedImm32(1), GPRInfo::regT1);
    jit.negPtr(GPRInfo::regT1, GPRInfo::regT4);
    jit.loadValue(MacroAssembler::BaseIndex(GPRInfo::regT0, GPRInfo::regT1, MacroAssembler::TimesEight, offsetOfLocals), JSRInfo::jsRegT32);
    jit.storeValue(JSRInfo::jsRegT32, MacroAssembler::BaseIndex(GPRInfo::callFrameRegister, GPRInfo::regT4, MacroAssembler::TimesEight, -static_cast<intptr_t>(sizeof(Register))));
    jit.branchPtr(MacroAssembler::NotEqual, GPRInfo::regT1, MacroAssembler::TrustedImmPtr(std::bit_cast<void*>(-static_cast<intptr_t>(CallFrame::headerSizeInRegisters)))).linkTo(loop, &jit);
    
    jit.loadPtr(MacroAssembler::Address(GPRInfo::regT0, offsetOfTargetPC), GPRInfo::regT1);
    MacroAssembler::Jump ok = jit.branchPtr(MacroAssembler::Above, GPRInfo::regT1, MacroAssembler::TrustedImmPtr(std::bit_cast<void*>(static_cast<intptr_t>(1000))));
    jit.abortWithReason(DFGUnreasonableOSREntryJumpDestination);

    ok.link(&jit);

    jit.jitAssertCodeBlockOnCallFrameIsOptimizingJIT(GPRInfo::regT2);

    jit.restoreCalleeSavesFromEntryFrameCalleeSavesBuffer(vm.topEntryFrame);
    jit.emitMaterializeTagCheckRegisters();
#if USE(JSVALUE64)
    jit.emitGetFromCallFrameHeaderPtr(CallFrameSlot::codeBlock, GPRInfo::jitDataRegister);
    jit.loadPtr(CCallHelpers::Address(GPRInfo::jitDataRegister, CodeBlock::offsetOfJITData()), GPRInfo::jitDataRegister);
#endif

    jit.farJump(GPRInfo::regT1, GPRInfo::callFrameRegister);

    LinkBuffer patchBuffer(jit, GLOBAL_THUNK_ID, LinkBuffer::Profile::DFGOSREntry);
    return FINALIZE_THUNK(patchBuffer, JITThunkPtrTag, nullptr, "DFG OSR entry thunk");
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
