/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

//===- MCMachOSymbolFlags.h - MachO Symbol Flags ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SymbolFlags used for the MachO target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMACHOSYMBOLFLAGS_H
#define LLVM_MC_MCMACHOSYMBOLFLAGS_H

// These flags are mostly used in MCMachOStreamer.cpp but also needed in
// MachObjectWriter.cpp to test for Weak Definitions of symbols to emit
// the correct relocation information.

namespace llvm {
  /// SymbolFlags - We store the value for the 'desc' symbol field in the lowest
  /// 16 bits of the implementation defined flags.
  enum SymbolFlags { // See <mach-o/nlist.h>.
    SF_DescFlagsMask                        = 0xFFFF,

    // Reference type flags.
    SF_ReferenceTypeMask                    = 0x0007,
    SF_ReferenceTypeUndefinedNonLazy        = 0x0000,
    SF_ReferenceTypeUndefinedLazy           = 0x0001,
    SF_ReferenceTypeDefined                 = 0x0002,
    SF_ReferenceTypePrivateDefined          = 0x0003,
    SF_ReferenceTypePrivateUndefinedNonLazy = 0x0004,
    SF_ReferenceTypePrivateUndefinedLazy    = 0x0005,

    // Other 'desc' flags.
    SF_ThumbFunc                            = 0x0008,
    SF_NoDeadStrip                          = 0x0020,
    SF_WeakReference                        = 0x0040,
    SF_WeakDefinition                       = 0x0080,
    SF_SymbolResolver                       = 0x0100
  };

} // end namespace llvm

#endif
