/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

namespace WTF {

#if defined(NDEBUG) \
    && (CPU(X86_64) || CPU(X86) || CPU(ARM64) || CPU(ARM_THUMB2) || CPU(ARM_TRADITIONAL))

// We can only use the inline asm implementation on release builds because it
// needs to be inlinable in order to be correct.
ALWAYS_INLINE void* currentStackPointer()
{
    void* stackPointer = nullptr;
#if CPU(X86_64)
    __asm__ volatile ("movq %%rsp, %0" : "=r"(stackPointer) ::);
#elif CPU(X86)
    __asm__ volatile ("movl %%esp, %0" : "=r"(stackPointer) ::);
#elif CPU(ARM64) && defined(__ILP32__)
    uint64_t stackPointerRegister = 0;
    __asm__ volatile ("mov %0, sp" : "=r"(stackPointerRegister) ::);
    stackPointer = reinterpret_cast<void*>(stackPointerRegister);
#elif CPU(ARM64) || CPU(ARM_THUMB2) || CPU(ARM_TRADITIONAL)
    __asm__ volatile ("mov %0, sp" : "=r"(stackPointer) ::);
#endif
    return stackPointer;
}

#elif !ENABLE(C_LOOP)

#define USE_ASM_CURRENT_STACK_POINTER 1
extern "C" WTF_EXPORT_PRIVATE void* CDECL currentStackPointer(void);

#else

#define USE_GENERIC_CURRENT_STACK_POINTER 1
WTF_EXPORT_PRIVATE void* currentStackPointer();

#endif

} // namespace WTF

using WTF::currentStackPointer;
