/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include <wtf/PlatformRegisters.h>

#include <wtf/PtrTag.h>

namespace WTF {

#if USE(PLATFORM_REGISTERS_WITH_PROFILE) && CPU(ARM64E)

#define USE_UNTAGGED_THREAD_STATE_PTR 1

void* threadStateLRInternal(PlatformRegisters& regs)
{
    void* candidateLR = arm_thread_state64_get_lr_fptr(regs);

#if USE(UNTAGGED_THREAD_STATE_PTR)
    if (candidateLR && isTaggedWith<CFunctionPtrTag>(candidateLR))
        return retagCodePtr<CFunctionPtrTag, PlatformRegistersLRPtrTag>(candidateLR);
    candidateLR = std::bit_cast<void*>(arm_thread_state64_get_lr(regs));
    if (!candidateLR)
        return candidateLR;
    return tagCodePtr<PlatformRegistersLRPtrTag>(candidateLR);

#else
    return retagCodePtr<CFunctionPtrTag, PlatformRegistersLRPtrTag>(candidateLR);
#endif
}

void* threadStatePCInternal(PlatformRegisters& regs)
{
#if CPU(ARM64E) && HAVE(HARDENED_MACH_EXCEPTIONS)
    // If we have modified the PC and set it to a presigned function we want to avoid
    // authing the value as it is using a custom ptrauth signing scheme.
    _STRUCT_ARM_THREAD_STATE64* ts = &(regs);
    if (!(ts->__opaque_flags & __DARWIN_ARM_THREAD_STATE64_FLAGS_KERNEL_SIGNED_PC))
        return nullptr;
#endif // CPU(ARM64E) && HAVE(HARDENED_MACH_EXCEPTIONS)

    void* candidatePC = arm_thread_state64_get_pc_fptr(regs);

#if USE(UNTAGGED_THREAD_STATE_PTR)
    if (candidatePC && isTaggedWith<CFunctionPtrTag>(candidatePC))
        return retagCodePtr<CFunctionPtrTag, PlatformRegistersPCPtrTag>(candidatePC);
    candidatePC = std::bit_cast<void*>(arm_thread_state64_get_pc(regs));
    if (!candidatePC)
        return candidatePC;
    return tagCodePtr<PlatformRegistersPCPtrTag>(candidatePC);

#else
    return retagCodePtr<CFunctionPtrTag, PlatformRegistersPCPtrTag>(candidatePC);
#endif
}

#endif // USE(PLATFORM_REGISTERS_WITH_PROFILE) && CPU(ARM64E)

} // namespace WTF
