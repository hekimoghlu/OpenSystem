/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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

//===-- PPCFixupKinds.h - PPC Specific Fixup Entries ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PPC_PPCFIXUPKINDS_H
#define LLVM_PPC_PPCFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace PPC {
enum Fixups {
  // fixup_ppc_br24 - 24-bit PC relative relocation for direct branches like 'b'
  // and 'bl'.
  fixup_ppc_br24 = FirstTargetFixupKind,
  
  /// fixup_ppc_brcond14 - 14-bit PC relative relocation for conditional
  /// branches.
  fixup_ppc_brcond14,
  
  /// fixup_ppc_lo16 - A 16-bit fixup corresponding to lo16(_foo) for instrs
  /// like 'li'.
  fixup_ppc_lo16,
  
  /// fixup_ppc_ha16 - A 16-bit fixup corresponding to ha16(_foo) for instrs
  /// like 'lis'.
  fixup_ppc_ha16,
  
  /// fixup_ppc_lo14 - A 14-bit fixup corresponding to lo16(_foo) for instrs
  /// like 'std'.
  fixup_ppc_lo14,
  
  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif
