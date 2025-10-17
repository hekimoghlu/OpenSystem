/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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

int puts(const char *p);

__attribute__((naked))
__attribute__((section(".start")))
void start() {
  asm volatile("la sp, stack + 8192 - 4");
  asm volatile("call main");
  asm volatile("call halt");
}

void halt(void) {
  puts("HALT\n");
  asm("unimp");
}

__attribute__((aligned(4))) char stack[8192];

int putchar(int c) {
  // This is only valid in an emulator (QEMU), and it's skipping a proper configuration of the UART device
  // and waiting for a "ready to transit" state.

  // QEMU riscv32-virt's specific location of the 16550A UART and its THR register
  *(volatile uint8_t *)(0x10000000 + 0) = c;
  return c;
}
