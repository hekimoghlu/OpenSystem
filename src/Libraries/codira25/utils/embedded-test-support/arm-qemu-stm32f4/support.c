/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift open source project
//
// Copyright (c) 2024 Apple Inc. and the Swift project authors.
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://language.org/LICENSE.txt for license information
// See https://language.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdint.h>

int main(int argc, char *argv[]);
void qemu_exit(void);
int puts(const char *);

__attribute__((noreturn))
void reset(void) {
  main(0, NULL);
  qemu_exit();
  __builtin_trap();
}

void interrupt(void) {
  puts("INTERRUPT\n");
  qemu_exit();
  while (1) {
  }
}

__attribute__((aligned(4))) char stack[8192];

__attribute((used))
__attribute((section(".vectors"))) void *vector_table[114] = {
    (void *)&stack[8192 - 4],  // initial SP
    reset,                 // Reset

    interrupt,  // NMI
    interrupt,  // HardFault
    interrupt,  // MemManage
    interrupt,  // BusFault
    interrupt,  // UsageFault

    0  // NULL for all the other handlers
};

void qemu_exit() {
  __asm__ volatile("mov r0, #0x18");
  __asm__ volatile("movw r1, #0x0026");
  __asm__ volatile("movt r1, #0x2"); // construct 0x20026 in r1
  __asm__ volatile("bkpt #0xab");
}

int putchar(int c) {
  // This is only valid in an emulator (QEMU), and it's skipping a proper configuration of the UART device
  // and waiting for a "ready to transit" state.

  // STM32F4 specific location of USART1 and its DR register
  *(volatile uint32_t *)(0x40011000 + 0x04) = c;
  return c;
}
