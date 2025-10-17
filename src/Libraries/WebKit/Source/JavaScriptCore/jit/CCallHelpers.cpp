/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
#include "CCallHelpers.h"

#if ENABLE(JIT)

#include "LinkBuffer.h"
#include "MaxFrameExtentForSlowPathCall.h"
#include "ShadowChicken.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CCallHelpers);

void CCallHelpers::logShadowChickenProloguePacket(GPRReg shadowPacket, GPRReg scratch1, GPRReg scope)
{
    storePtr(GPRInfo::callFrameRegister, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, frame)));
    loadPtr(Address(GPRInfo::callFrameRegister, OBJECT_OFFSETOF(CallerFrameAndPC, callerFrame)), scratch1);
    storePtr(scratch1, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, callerFrame)));
    loadPtr(addressFor(CallFrameSlot::callee), scratch1);
    storePtr(scratch1, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, callee)));
    storePtr(scope, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, scope)));
}

void CCallHelpers::ensureShadowChickenPacket(VM& vm, GPRReg shadowPacket, GPRReg scratch1NonArgGPR, GPRReg scratch2)
{
    ShadowChicken* shadowChicken = vm.shadowChicken();
    RELEASE_ASSERT(shadowChicken);
    ASSERT(!RegisterSetBuilder::argumentGPRs().contains(scratch1NonArgGPR, IgnoreVectors));
    move(TrustedImmPtr(shadowChicken->addressOfLogCursor()), scratch1NonArgGPR);
    loadPtr(Address(scratch1NonArgGPR), shadowPacket);
    Jump ok = branchPtr(Below, shadowPacket, TrustedImmPtr(shadowChicken->logEnd()));
    setupArguments<decltype(operationProcessShadowChickenLog)>(TrustedImmPtr(&vm));
    prepareCallOperation(vm);
    move(TrustedImmPtr(tagCFunction<OperationPtrTag>(operationProcessShadowChickenLog)), scratch1NonArgGPR);
    call(scratch1NonArgGPR, OperationPtrTag);
    move(TrustedImmPtr(shadowChicken->addressOfLogCursor()), scratch1NonArgGPR);
    loadPtr(Address(scratch1NonArgGPR), shadowPacket);
    ok.link(this);
    addPtr(TrustedImm32(sizeof(ShadowChicken::Packet)), shadowPacket, scratch2);
    storePtr(scratch2, Address(scratch1NonArgGPR));
}


template <typename CodeBlockType>
void CCallHelpers::logShadowChickenTailPacketImpl(GPRReg shadowPacket, JSValueRegs thisRegs, GPRReg scope, CodeBlockType codeBlock, CallSiteIndex callSiteIndex)
{
    storePtr(GPRInfo::callFrameRegister, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, frame)));
    storePtr(TrustedImmPtr(ShadowChicken::Packet::tailMarker()), Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, callee)));
    storeValue(thisRegs, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, thisValue)));
    storePtr(scope, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, scope)));
    storePtr(codeBlock, Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, codeBlock)));
    store32(TrustedImm32(callSiteIndex.bits()), Address(shadowPacket, OBJECT_OFFSETOF(ShadowChicken::Packet, callSiteIndex)));
}

void CCallHelpers::logShadowChickenTailPacket(GPRReg shadowPacket, JSValueRegs thisRegs, GPRReg scope, GPRReg codeBlock, CallSiteIndex callSiteIndex)
{
    logShadowChickenTailPacketImpl(shadowPacket, thisRegs, scope, codeBlock, callSiteIndex);
}

static_assert(!((maxFrameExtentForSlowPathCall + 2 * sizeof(CPURegister)) % 16), "Stack must be aligned after CTI thunk entry");

void CCallHelpers::emitCTIThunkPrologue(bool returnAddressAlreadyTagged)
{
    // Stash frame pointer and return address
    if (!returnAddressAlreadyTagged)
        tagReturnAddress();
#if CPU(X86_64)
    push(X86Registers::ebp); // return address pushed by the call instruction
#elif CPU(ARM64) || CPU(ARM_THUMB2) || CPU(RISCV64)
    pushPair(framePointerRegister, linkRegister);
#else
#   error "Not implemented on platform"
#endif
    // Make enough space on the stack to pass arguments in a call
    if constexpr (!!maxFrameExtentForSlowPathCall)
        subPtr(TrustedImm32(maxFrameExtentForSlowPathCall), stackPointerRegister);
}

void CCallHelpers::emitCTIThunkEpilogue()
{
    // Reset stack
    if constexpr (!!maxFrameExtentForSlowPathCall)
        addPtr(TrustedImm32(maxFrameExtentForSlowPathCall), stackPointerRegister);
    // Restore frame pointer and return address
#if CPU(X86_64)
    pop(X86Registers::ebp); // Return address left on stack
#elif CPU(ARM64) || CPU(ARM_THUMB2) || CPU(RISCV64)
    popPair(framePointerRegister, linkRegister);
#else
#   error "Not implemented on platform"
#endif
}

} // namespace JSC

#endif // ENABLE(JIT)
