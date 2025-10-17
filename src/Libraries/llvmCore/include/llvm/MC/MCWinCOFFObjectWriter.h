/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

//===-- llvm/MC/MCWinCOFFObjectWriter.h - Win COFF Object Writer *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWINCOFFOBJECTWRITER_H
#define LLVM_MC_MCWINCOFFOBJECTWRITER_H

namespace llvm {
  class MCWinCOFFObjectTargetWriter {
    const unsigned Machine;

  protected:
    MCWinCOFFObjectTargetWriter(unsigned Machine_);

  public:
    virtual ~MCWinCOFFObjectTargetWriter() {}

    unsigned getMachine() const { return Machine; }
    virtual unsigned getRelocType(unsigned FixupKind) const = 0;
  };

  /// \brief Construct a new Win COFF writer instance.
  ///
  /// \param MOTW - The target specific WinCOFF writer subclass.
  /// \param OS - The stream to write to.
  /// \returns The constructed object writer.
  MCObjectWriter *createWinCOFFObjectWriter(MCWinCOFFObjectTargetWriter *MOTW,
                                            raw_ostream &OS);
} // End llvm namespace

#endif
