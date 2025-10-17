/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#include "../../bionic/libc_init_common.h"
#include <stddef.h>
#include <stdint.h>

extern init_func_t* __preinit_array_start[];
extern init_func_t* __preinit_array_end[];
extern init_func_t* __init_array_start[];
extern init_func_t* __init_array_end[];
extern fini_func_t* __fini_array_start[];
extern fini_func_t* __fini_array_end[];

#if !defined(CRTBEGIN_STATIC)
/* This function will be called during normal program termination
 * to run the destructors that are listed in the .fini_array section
 * of the executable, if any.
 *
 * 'fini_array' points to a list of function addresses.
 */
static void call_fini_array() {
  fini_func_t** array = __fini_array_start;
  size_t count = __fini_array_end - __fini_array_start;
  // Call fini functions in reverse order.
  while (count-- > 0) {
    fini_func_t* function = array[count];
    (*function)();
  }
}

// libc.so needs fini_array with sentinels. So create a fake fini_array with sentinels.
// It contains a function to call functions in real fini_array.
static fini_func_t* fini_array_with_sentinels[] = {
    (fini_func_t*)-1,
    &call_fini_array,
    (fini_func_t*)0,
};
#endif  // !defined(CRTBEGIN_STATIC)

__used static void _start_main(void* raw_args) {
  structors_array_t array = {};
#if defined(CRTBEGIN_STATIC)
  array.preinit_array = __preinit_array_start;
  array.preinit_array_count = __preinit_array_end - __preinit_array_start;
  array.init_array = __init_array_start;
  array.init_array_count = __init_array_end - __init_array_start;
  array.fini_array = __fini_array_start;
  array.fini_array_count = __fini_array_end - __fini_array_start;
#else
  if (__fini_array_end - __fini_array_start > 0) {
    array.fini_array = fini_array_with_sentinels;
  }
#endif  // !defined(CRTBEGIN_STATIC)

  __libc_init(raw_args, NULL, &main, &array);
}

#define PRE ".text; .global _start; .type _start,%function; _start:"
#define POST "; .size _start, .-_start"

#if defined(__aarch64__)
__asm__(PRE "bti j; mov x29,#0; mov x30,#0; mov x0,sp; b _start_main" POST);
#elif defined(__arm__)
__asm__(PRE "mov fp,#0; mov lr,#0; mov r0,sp; b _start_main" POST);
#elif defined(__i386__)
__asm__(PRE
        "xorl %ebp,%ebp; movl %esp,%eax; andl $~0xf,%esp; subl $12,%esp; pushl %eax;"
        "call _start_main" POST);
#elif defined(__riscv)
__asm__(PRE "li fp,0; li ra,0; mv a0,sp; tail _start_main" POST);
#elif defined(__x86_64__)
__asm__(PRE "xorl %ebp, %ebp; movq %rsp,%rdi; andq $~0xf,%rsp; callq _start_main" POST);
#else
#error unsupported architecture
#endif

#undef PRE
#undef POST

// On arm32 and arm64, when targeting Q and up, overalign the TLS segment to
// (8 * sizeof(void*)), which reserves enough space between the thread pointer
// and the executable's TLS segment for Bionic's TLS slots. It has the side
// effect of placing a 0-sized TLS segment into Android executables that don't
// use TLS, but this should be harmless.
//
// To ensure that the .tdata input section isn't deleted (e.g. by
// --gc-sections), the .text input section (which contains _start) has a
// relocation to the .tdata input section.
#if __ANDROID_API__ >= 29
#if defined(__arm__)
asm("  .section .tdata,\"awT\",%progbits\n"
    "  .p2align 5\n"
    "  .text\n"
    "  .reloc 0, R_ARM_NONE, .tdata\n");
#elif defined(__aarch64__)
asm("  .section .tdata,\"awT\",@progbits\n"
    "  .p2align 6\n"
    "  .text\n"
    "  .reloc 0, R_AARCH64_NONE, .tdata\n");
#endif
#endif

#include "__dso_handle.h"
#include "atexit.h"
#include "pthread_atfork.h"
