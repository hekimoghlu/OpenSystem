/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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

#if defined(__riscv)
// TLS_DTV_OFFSET is a constant used in relocation fields, defined in RISC-V ELF Specification[1]
// The front of the TCB contains a pointer to the DTV, and each pointer in DTV
// points to 0x800 past the start of a TLS block to make full use of the range
// of load/store instructions, refer to [2].
//
// [1]: RISC-V ELF Specification.
// https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#constants
// [2]: Documentation of TLS data structures
// https://github.com/riscv-non-isa/riscv-elf-psabi-doc/issues/53
#define TLS_DTV_OFFSET 0x800
#else
#define TLS_DTV_OFFSET 0
#endif
