/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
#include "StackPointer.h"

#include "InlineASM.h"

namespace WTF {

#if USE(ASM_CURRENT_STACK_POINTER)

#if CPU(X86) && COMPILER(MSVC)
extern "C" __declspec(naked) void currentStackPointer()
{
    __asm {
        mov eax, esp
        add eax, 4
        ret
    }
}

#elif CPU(X86)
asm (
    ".text" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"
    "movl %esp, %eax" "\n"
    "addl $4, %eax" "\n"
    "ret" "\n"
    ".previous" "\n"
);

#elif CPU(X86_64) && OS(WINDOWS)

// The Win64 port will use a hack where we define currentStackPointer in
// LowLevelInterpreter.asm.

asm (
    ".text" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"

    "movq %rsp, %rax" "\n"
    "addq $40, %rax" "\n" // Account for return address and shadow stack
    "ret" "\n"

    ".section .drectve" "\n"
    ".ascii \"-export:currentStackPointer\"" "\n"
);

#elif CPU(X86_64)
asm (
    ".text" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"

    "movq  %rsp, %rax" "\n"
    "addq $8, %rax" "\n" // Account for return address.
    "ret" "\n"
    ".previous" "\n"
);

#elif CPU(ARM64E)
asm (
    ".text" "\n"
    ".balign 16" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"

    "pacibsp" "\n"
    "mov x0, sp" "\n"
    "retab" "\n"
    ".previous" "\n"
);

#elif CPU(ARM64)
asm (
    ".text" "\n"
    ".balign 16" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"

    "mov x0, sp" "\n"
    "ret" "\n"
    ".previous" "\n"
);

#elif CPU(ARM_THUMB2)
asm (
    ".text" "\n"
    ".align 2" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    ".thumb" "\n"
    ".thumb_func " THUMB_FUNC_PARAM(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"

    "mov r0, sp" "\n"
    "bx  lr" "\n"
    ".previous" "\n"
);

#elif CPU(MIPS)
asm (
    ".text" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"
    ".set push" "\n"
    ".set noreorder" "\n"
    ".set noat" "\n"

    "move $v0, $sp" "\n"
    "jr   $ra" "\n"
    "nop" "\n"
    ".set pop" "\n"
    ".previous" "\n"
);

#elif CPU(RISCV64)
asm (
    ".text" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"

     "mv x10, sp" "\n"
     "ret" "\n"
     ".previous" "\n"
);

#elif CPU(LOONGARCH64)
asm (
    ".text" "\n"
    ".globl " SYMBOL_STRING(currentStackPointer) "\n"
    SYMBOL_STRING(currentStackPointer) ":" "\n"

     "move $r4, $r3" "\n"
     "jr   $r1" "\n"
     ".previous" "\n"
);

#else
#error "Unsupported platform: need implementation of currentStackPointer."
#endif

#elif USE(GENERIC_CURRENT_STACK_POINTER)
constexpr size_t sizeOfFrameHeader = 2 * sizeof(void*);

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

SUPPRESS_ASAN NEVER_INLINE
void* currentStackPointer()
{
    return reinterpret_cast<uint8_t*>(__builtin_frame_address(0)) + sizeOfFrameHeader;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
#endif // USE(GENERIC_CURRENT_STACK_POINTER)

} // namespace WTF
