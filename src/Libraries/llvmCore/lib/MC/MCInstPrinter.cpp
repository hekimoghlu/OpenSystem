/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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

//===-- MCInstPrinter.cpp - Convert an MCInst to target assembly syntax ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCInstPrinter::~MCInstPrinter() {
}

/// getOpcodeName - Return the name of the specified opcode enum (e.g.
/// "MOV32ri") or empty if we can't resolve it.
StringRef MCInstPrinter::getOpcodeName(unsigned Opcode) const {
  return MII.getName(Opcode);
}

void MCInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  llvm_unreachable("Target should implement this");
}

void MCInstPrinter::printAnnotation(raw_ostream &OS, StringRef Annot) {
  if (!Annot.empty()) {
    if (CommentStream)
      (*CommentStream) << Annot;
    else
      OS << " " << MAI.getCommentString() << " " << Annot;
  }
}
