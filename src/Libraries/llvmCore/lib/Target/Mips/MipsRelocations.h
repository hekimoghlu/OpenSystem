/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

//===-- MipsRelocations.h - Mips Code Relocations ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Mips target-specific relocation types
// (for relocation-model=static).
//
//===----------------------------------------------------------------------===//

#ifndef MIPSRELOCATIONS_H_
#define MIPSRELOCATIONS_H_

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace Mips{
    enum RelocationType {
      // reloc_mips_pc16 - pc relative relocation for branches. The lower 18
      // bits of the difference between the branch target and the branch
      // instruction, shifted right by 2.
      reloc_mips_pc16 = 1,

      // reloc_mips_hi - upper 16 bits of the address (modified by +1 if the
      // lower 16 bits of the address is negative).
      reloc_mips_hi = 2,

      // reloc_mips_lo - lower 16 bits of the address.
      reloc_mips_lo = 3,

      // reloc_mips_26 - lower 28 bits of the address, shifted right by 2.
      reloc_mips_26 = 4
    };
  }
}

#endif /* MIPSRELOCATIONS_H_ */
