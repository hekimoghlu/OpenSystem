/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
// clang interprets -fno-builtin more loosely than you might expect,
// and thinks it's okay to still substitute builtins as long as they're
// named __aeabi_* rather than __builtin_*, which causes infinite
// recursion if we have the fortified memcpy visible in this file.
#undef _FORTIFY_SOURCE

#include <stddef.h>
#include <string.h>

extern int __cxa_atexit(void (*)(void*), void*, void*);

// All of these are weak symbols to avoid multiple definition errors when
// linking with libstdc++-v3 or compiler-rt.

/* The "C++ ABI for ARM" document states that static C++ constructors,
 * which are called from the .init_array, should manually call
 * __aeabi_atexit() to register static destructors explicitly.
 *
 * Note that 'dso_handle' is the address of a magic linker-generate
 * variable from the shared object that contains the constructor/destructor
 */

int __attribute__((weak))
__aeabi_atexit_impl(void *object, void (*destructor) (void *), void *dso_handle) {
    return __cxa_atexit(destructor, object, dso_handle);
}

int __attribute__((weak))
__aeabi_atexit_impl2(void *object, void (*destructor) (void *), void *dso_handle) {
    return __cxa_atexit(destructor, object, dso_handle);
}


void __attribute__((weak)) __aeabi_memcpy8_impl(void *dest, const void *src, size_t n) {
    memcpy(dest, src, n);
}

void __attribute__((weak)) __aeabi_memcpy4_impl(void *dest, const void *src, size_t n) {
    memcpy(dest, src, n);
}

void __attribute__((weak)) __aeabi_memcpy_impl(void *dest, const void *src, size_t n) {
    memcpy(dest, src, n);
}

void __attribute__((weak)) __aeabi_memcpy8_impl2(void *dest, const void *src, size_t n) {
    memcpy(dest, src, n);
}

void __attribute__((weak)) __aeabi_memcpy4_impl2(void *dest, const void *src, size_t n) {
    memcpy(dest, src, n);
}

void __attribute__((weak)) __aeabi_memcpy_impl2(void *dest, const void *src, size_t n) {
    memcpy(dest, src, n);
}


void __attribute__((weak)) __aeabi_memmove8_impl(void *dest, const void *src, size_t n) {
    memmove(dest, src, n);
}

void __attribute__((weak)) __aeabi_memmove4_impl(void *dest, const void *src, size_t n) {
    memmove(dest, src, n);
}

void __attribute__((weak)) __aeabi_memmove_impl(void *dest, const void *src, size_t n) {
    memmove(dest, src, n);
}

void __attribute__((weak)) __aeabi_memmove8_impl2(void *dest, const void *src, size_t n) {
    memmove(dest, src, n);
}

void __attribute__((weak)) __aeabi_memmove4_impl2(void *dest, const void *src, size_t n) {
    memmove(dest, src, n);
}

void __attribute__((weak)) __aeabi_memmove_impl2(void *dest, const void *src, size_t n) {
    memmove(dest, src, n);
}

/*
 * __aeabi_memset has the order of its second and third arguments reversed.
 *  This allows __aeabi_memclr to tail-call __aeabi_memset
 */

void __attribute__((weak)) __aeabi_memset8_impl(void *dest, size_t n, int c) {
    memset(dest, c, n);
}

void __attribute__((weak)) __aeabi_memset4_impl(void *dest, size_t n, int c) {
    memset(dest, c, n);
}

void __attribute__((weak)) __aeabi_memset_impl(void *dest, size_t n, int c) {
    memset(dest, c, n);
}

void __attribute__((weak)) __aeabi_memset8_impl2(void *dest, size_t n, int c) {
    memset(dest, c, n);
}

void __attribute__((weak)) __aeabi_memset4_impl2(void *dest, size_t n, int c) {
    memset(dest, c, n);
}

void __attribute__((weak)) __aeabi_memset_impl2(void *dest, size_t n, int c) {
    memset(dest, c, n);
}


void __attribute__((weak)) __aeabi_memclr8_impl(void *dest, size_t n) {
    __aeabi_memset8_impl(dest, n, 0);
}

void __attribute__((weak)) __aeabi_memclr4_impl(void *dest, size_t n) {
    __aeabi_memset4_impl(dest, n, 0);
}

void __attribute__((weak)) __aeabi_memclr_impl(void *dest, size_t n) {
    __aeabi_memset_impl(dest, n, 0);
}

void __attribute__((weak)) __aeabi_memclr8_impl2(void *dest, size_t n) {
    __aeabi_memset8_impl(dest, n, 0);
}

void __attribute__((weak)) __aeabi_memclr4_impl2(void *dest, size_t n) {
    __aeabi_memset4_impl(dest, n, 0);
}

void __attribute__((weak)) __aeabi_memclr_impl2(void *dest, size_t n) {
    __aeabi_memset_impl(dest, n, 0);
}

#define __AEABI_SYMVERS(fn_name) \
__asm__(".symver " #fn_name "_impl, " #fn_name "@@LIBC_N"); \
__asm__(".symver " #fn_name "_impl2, " #fn_name "@LIBC_PRIVATE")

__AEABI_SYMVERS(__aeabi_atexit);
__AEABI_SYMVERS(__aeabi_memcpy8);
__AEABI_SYMVERS(__aeabi_memcpy4);
__AEABI_SYMVERS(__aeabi_memcpy);
__AEABI_SYMVERS(__aeabi_memmove8);
__AEABI_SYMVERS(__aeabi_memmove4);
__AEABI_SYMVERS(__aeabi_memmove);
__AEABI_SYMVERS(__aeabi_memset8);
__AEABI_SYMVERS(__aeabi_memset4);
__AEABI_SYMVERS(__aeabi_memset);
__AEABI_SYMVERS(__aeabi_memclr8);
__AEABI_SYMVERS(__aeabi_memclr4);
__AEABI_SYMVERS(__aeabi_memclr);

#undef __AEABI_SYMVERS
