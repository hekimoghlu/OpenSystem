/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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

/* https://github.com/android/ndk/issues/1422 */
#include <features.h>

#include <asm/unistd.h> /* For system call numbers. */
#define MAX_ERRNO 4095  /* For recognizing system call error returns. */

#define __bionic_asm_custom_entry(f)
#define __bionic_asm_custom_end(f)
#define __bionic_asm_function_type @function
#define __bionic_asm_custom_note_gnu_section()

#if defined(__aarch64__)
#include <private/bionic_asm_arm64.h>
#elif defined(__arm__)
#include <private/bionic_asm_arm.h>
#elif defined(__i386__)
#include <private/bionic_asm_x86.h>
#elif defined(__riscv)
#include <private/bionic_asm_riscv64.h>
#elif defined(__x86_64__)
#include <private/bionic_asm_x86_64.h>
#endif

// Starts a normal assembler routine.
#define ENTRY(__f) __ENTRY_WITH_BINDING(__f, .globl)

// Starts an assembler routine with hidden visibility.
#define ENTRY_PRIVATE(__f)           \
  __ENTRY_WITH_BINDING(__f, .globl); \
  .hidden __f;

// Starts an assembler routine that's weak so native bridges can override it.
#define ENTRY_WEAK_FOR_NATIVE_BRIDGE(__f) __ENTRY_WITH_BINDING(__f, .weak)

// Starts an assembler routine with hidden visibility and no DWARF information.
// Only used for internal functions passed via sa_restorer.
// TODO: can't we just delete all those and let the kernel do its thing?
#define ENTRY_NO_DWARF_PRIVATE(__f) \
  __ENTRY_NO_DWARF(__f, .globl);    \
  .hidden __f;

// (Implementation detail.)
#define __ENTRY_NO_DWARF(__f, __binding) \
  .text;                                 \
  __binding __f;                         \
  .balign __bionic_asm_align;            \
  .type __f, __bionic_asm_function_type; \
  __f:                                   \
  __bionic_asm_custom_entry(__f);

// (Implementation detail.)
#define __ENTRY_WITH_BINDING(__f, __binding) \
  __ENTRY_NO_DWARF(__f, __binding);          \
  .cfi_startproc;

// Ends a normal assembler routine.
#define END(__f) \
  .cfi_endproc;  \
  END_NO_DWARF(__f)

// Ends an assembler routine with no DWARF information.
// Only used for internal functions passed via sa_restorer.
// TODO: can't we just delete all those and let the kernel do its thing?
#define END_NO_DWARF(__f) \
  .size __f, .- __f;      \
  __bionic_asm_custom_end(__f)

// Creates an alias `alias` for the symbol `original`.
#define ALIAS_SYMBOL(alias, original) \
  .globl alias;                       \
  .equ alias, original

// Creates an alias `alias` for the symbol `original` that's weak so it can be
// separately overridden by native bridges.
#define ALIAS_SYMBOL_WEAK_FOR_NATIVE_BRIDGE(alias, original) \
  .weak alias;                                               \
  .equ alias, original

// Adds a GNU property ELF note. Important on arm64 to declare PAC/BTI support.
#define NOTE_GNU_PROPERTY() __bionic_asm_custom_note_gnu_section()

// Gives local labels a more convenient and readable syntax.
#define L(__label) .L##__label
