/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

#include <wtf/Platform.h>
#include <wtf/StdLibExtras.h>

#if OS(DARWIN)
#include <mach/exception_types.h>
#include <mach/thread_act.h>
#include <signal.h>
#elif OS(WINDOWS)
#include <windows.h>
#elif OS(OPENBSD)
typedef ucontext_t mcontext_t;
#elif OS(QNX)
#include <ucontext.h>
#else
#include <sys/ucontext.h>
#endif

namespace WTF {

#if OS(DARWIN)

#if CPU(X86)
typedef i386_thread_state_t PlatformRegisters;
#elif CPU(X86_64)
typedef x86_thread_state64_t PlatformRegisters;
#elif CPU(PPC)
typedef ppc_thread_state_t PlatformRegisters;
#elif CPU(PPC64)
typedef ppc_thread_state64_t PlatformRegisters;
#elif CPU(ARM)
typedef arm_thread_state_t PlatformRegisters;
#elif CPU(ARM64)
typedef arm_thread_state64_t PlatformRegisters;
#else
#error Unknown Architecture
#endif

inline PlatformRegisters& registersFromUContext(ucontext_t* ucontext)
{
    return ucontext->uc_mcontext->__ss;
}

#elif OS(WINDOWS)

using PlatformRegisters = CONTEXT;

#elif HAVE(MACHINE_CONTEXT)

struct PlatformRegisters {
    mcontext_t machineContext;
};

inline PlatformRegisters& registersFromUContext(ucontext_t* ucontext)
{
#if OS(OPENBSD)
    return *std::bit_cast<PlatformRegisters*>(ucontext);
#elif CPU(PPC)
    return *std::bit_cast<PlatformRegisters*>(ucontext->uc_mcontext.uc_regs);
#else
    return *std::bit_cast<PlatformRegisters*>(&ucontext->uc_mcontext);
#endif
}

#else

struct PlatformRegisters {
    void* stackPointer;
};

#endif

} // namespace WTF

#if USE(PLATFORM_REGISTERS_WITH_PROFILE)
#if CPU(ARM64E)

namespace WTF {

extern void* threadStateLRInternal(PlatformRegisters&);
extern void* threadStatePCInternal(PlatformRegisters&);

} // namespace WTF

using WTF::threadStateLRInternal;
using WTF::threadStatePCInternal;

#else // not CPU(ARM64E)

#define threadStateLRInternal(regs) std::bit_cast<void*>(arm_thread_state64_get_lr(regs))
#define threadStatePCInternal(regs) std::bit_cast<void*>(arm_thread_state64_get_pc(regs))

#endif // CPU(ARM64E)

#define WTF_READ_PLATFORM_REGISTERS_SP_WITH_PROFILE(regs) \
    reinterpret_cast<void*>(arm_thread_state64_get_sp(const_cast<PlatformRegisters&>(regs)))

#define WTF_READ_PLATFORM_REGISTERS_FP_WITH_PROFILE(regs) \
    reinterpret_cast<void*>(arm_thread_state64_get_fp(const_cast<PlatformRegisters&>(regs)))

#define WTF_READ_PLATFORM_REGISTERS_LR_WITH_PROFILE(regs) \
    threadStateLRInternal(const_cast<PlatformRegisters&>(regs))

#define WTF_READ_PLATFORM_REGISTERS_PC_WITH_PROFILE(regs) \
    threadStatePCInternal(const_cast<PlatformRegisters&>(regs))

#if CPU(ARM64) && HAVE(HARDENED_MACH_EXCEPTIONS)
#define WTF_WRITE_PLATFORM_REGISTERS_PC_WITH_PROFILE(regs, newPointer) \
    arm_thread_state64_set_pc_presigned_fptr(regs, newPointer)
#else
#define WTF_WRITE_PLATFORM_REGISTERS_PC_WITH_PROFILE(regs, newPointer) \
    arm_thread_state64_set_pc_fptr(regs, newPointer)
#endif // CPU(ARM64) && HAVE(HARDENED_MACH_EXCEPTIONS)

#define WTF_READ_MACHINE_CONTEXT_SP_WITH_PROFILE(machineContext) \
    WTF_READ_PLATFORM_REGISTERS_SP_WITH_PROFILE(machineContext->__ss)

#define WTF_READ_MACHINE_CONTEXT_FP_WITH_PROFILE(machineContext) \
    WTF_READ_PLATFORM_REGISTERS_FP_WITH_PROFILE(machineContext->__ss)

#define WTF_READ_MACHINE_CONTEXT_PC_WITH_PROFILE(machineContext) \
    WTF_READ_PLATFORM_REGISTERS_PC_WITH_PROFILE(machineContext->__ss)

#endif // USE(PLATFORM_REGISTERS_WITH_PROFILE)

using WTF::PlatformRegisters;
