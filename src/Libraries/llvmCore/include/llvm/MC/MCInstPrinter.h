/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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

//===-- MCInstPrinter.h - Convert an MCInst to target assembly syntax -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINSTPRINTER_H
#define LLVM_MC_MCINSTPRINTER_H

namespace llvm {
class MCInst;
class raw_ostream;
class MCAsmInfo;
class MCInstrInfo;
class MCRegisterInfo;
class StringRef;

/// MCInstPrinter - This is an instance of a target assembly language printer
/// that converts an MCInst to valid target assembly syntax.
class MCInstPrinter {
protected:
  /// CommentStream - a stream that comments can be emitted to if desired.
  /// Each comment must end with a newline.  This will be null if verbose
  /// assembly emission is disable.
  raw_ostream *CommentStream;
  const MCAsmInfo &MAI;
  const MCInstrInfo &MII;
  const MCRegisterInfo &MRI;

  /// The current set of available features.
  unsigned AvailableFeatures;

  /// Utility function for printing annotations.
  void printAnnotation(raw_ostream &OS, StringRef Annot);
public:
  MCInstPrinter(const MCAsmInfo &mai, const MCInstrInfo &mii,
                const MCRegisterInfo &mri)
    : CommentStream(0), MAI(mai), MII(mii), MRI(mri), AvailableFeatures(0) {}

  virtual ~MCInstPrinter();

  /// setCommentStream - Specify a stream to emit comments to.
  void setCommentStream(raw_ostream &OS) { CommentStream = &OS; }

  /// printInst - Print the specified MCInst to the specified raw_ostream.
  ///
  virtual void printInst(const MCInst *MI, raw_ostream &OS,
                         StringRef Annot) = 0;

  /// getOpcodeName - Return the name of the specified opcode enum (e.g.
  /// "MOV32ri") or empty if we can't resolve it.
  StringRef getOpcodeName(unsigned Opcode) const;

  /// printRegName - Print the assembler register name.
  virtual void printRegName(raw_ostream &OS, unsigned RegNo) const;

  unsigned getAvailableFeatures() const { return AvailableFeatures; }
  void setAvailableFeatures(unsigned Value) { AvailableFeatures = Value; }
};

} // namespace llvm

#endif
