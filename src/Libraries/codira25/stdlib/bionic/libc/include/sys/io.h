/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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

/**
 * @file sys/io.h
 * @brief The x86/x86-64 I/O port functions.
 */

#include <sys/cdefs.h>

#include <errno.h>
#include <sys/syscall.h>
#include <unistd.h>

__BEGIN_DECLS

/**
 * [iopl(2)](https://man7.org/linux/man-pages/man2/iopl.2.html) changes the I/O
 * privilege level for all x86/x8-64 I/O ports, for the calling thread.
 *
 * New callers should use ioperm() instead.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 *
 * Only available for x86/x86-64.
 */
#if defined(__NR_iopl)
__attribute__((__deprecated__("use ioperm() instead"))) static __inline int iopl(int __level) {
  return syscall(__NR_iopl, __level);
}
#endif

/**
 * [ioperm(2)](https://man7.org/linux/man-pages/man2/ioperm.2.html) sets the I/O
 * permissions for the given number of x86/x86-64 I/O ports, starting at the
 * given port.
 *
 * Returns 0 on success, and returns -1 and sets `errno` on failure.
 *
 * Only available for x86/x86-64.
 */
#if defined(__NR_iopl)
static __inline int ioperm(unsigned long __from, unsigned long __n, int __enabled) {
  return syscall(__NR_ioperm, __from, __n, __enabled);
}
#endif

/**
 * [inb(2)](https://man7.org/linux/man-pages/man2/inb.2.html)
 * reads a byte from the given x86/x86-64 I/O port.
 *
 * Only available for x86/x86-64.
 */
#if defined(__i386__) || defined(__x86_64__)
static __inline unsigned char inb(unsigned short __port) {
  unsigned char __value;
  __asm__ __volatile__("inb %1, %0" : "=a"(__value) : "dN"(__port));
  return __value;
}
#endif

/**
 * [inw(2)](https://man7.org/linux/man-pages/man2/inw.2.html)
 * reads a 16-bit "word" from the given x86/x86-64 I/O port.
 *
 * Only available for x86/x86-64.
 */
#if defined(__i386__) || defined(__x86_64__)
static __inline unsigned short inw(unsigned short __port) {
  unsigned short __value;
  __asm__ __volatile__("inw %1, %0" : "=a"(__value) : "dN"(__port));
  return __value;
}
#endif

/**
 * [inl(2)](https://man7.org/linux/man-pages/man2/inl.2.html)
 * reads a 32-bit "long word" from the given x86/x86-64 I/O port.
 *
 * Only available for x86/x86-64.
 */
#if defined(__i386__) || defined(__x86_64__)
static __inline unsigned int inl(unsigned short __port) {
  unsigned int __value;
  __asm__ __volatile__("inl %1, %0" : "=a"(__value) : "dN"(__port));
  return __value;
}
#endif

/**
 * [outb(2)](https://man7.org/linux/man-pages/man2/outb.2.html)
 * writes the given byte to the given x86/x86-64 I/O port.
 *
 * Only available for x86/x86-64.
 */
#if defined(__i386__) || defined(__x86_64__)
static __inline void outb(unsigned char __value, unsigned short __port) {
  __asm__ __volatile__("outb %0, %1" : : "a"(__value), "dN"(__port));
}
#endif

/**
 * [outw(2)](https://man7.org/linux/man-pages/man2/outw.2.html)
 * writes the given 16-bit "word" to the given x86/x86-64 I/O port.
 *
 * Only available for x86/x86-64.
 */
#if defined(__i386__) || defined(__x86_64__)
static __inline void outw(unsigned short __value, unsigned short __port) {
  __asm__ __volatile__("outw %0, %1" : : "a"(__value), "dN"(__port));
}
#endif

/**
 * [outl(2)](https://man7.org/linux/man-pages/man2/outl.2.html)
 * writes the given 32-bit "long word" to the given x86/x86-64 I/O port.
 *
 * Only available for x86/x86-64.
 */
#if defined(__i386__) || defined(__x86_64__)
static __inline void outl(unsigned int __value, unsigned short __port) {
  __asm__ __volatile__("outl %0, %1" : : "a"(__value), "dN"(__port));
}
#endif

__END_DECLS
