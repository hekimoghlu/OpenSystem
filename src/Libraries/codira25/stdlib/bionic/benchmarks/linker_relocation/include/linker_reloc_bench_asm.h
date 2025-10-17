/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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

#if defined(__arm__)

#define GOT_RELOC(sym) .long sym(GOT_PREL)
#define CALL(sym) bl sym
#define DATA_WORD(val) .long val
#define MAIN .globl main; main: mov r0, #0; bx lr

#elif defined(__aarch64__)

#define GOT_RELOC(sym) adrp x1, :got:sym
#define CALL(sym) bl sym
#define DATA_WORD(val) .quad val
#define MAIN .globl main; main: mov w0, wzr; ret

#elif defined(__riscv)

#define GOT_RELOC(sym) lga a0, sym
#define CALL(sym) call sym@plt
#define DATA_WORD(val) .quad val
#define MAIN .globl main; main: li a0, 0; ret

#elif defined(__i386__)

#define GOT_RELOC(sym) .long sym@got
#define CALL(sym) call sym@PLT
#define DATA_WORD(val) .long val
#define MAIN .globl main; main: xorl %eax, %eax; retl

#elif defined(__x86_64__)

#define GOT_RELOC(sym) .quad sym@got
#define CALL(sym) call sym@PLT
#define DATA_WORD(val) .quad val
#define MAIN .globl main; main: xorl %eax, %eax; retq

#else
#error "Unrecognized architecture"
#endif
