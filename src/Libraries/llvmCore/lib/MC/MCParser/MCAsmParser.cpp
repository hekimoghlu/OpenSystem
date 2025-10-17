/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

//===-- MCAsmParser.cpp - Abstract Asm Parser Interface -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

MCAsmParser::MCAsmParser() : TargetParser(0), ShowParsedOperands(0) {
}

MCAsmParser::~MCAsmParser() {
}

void MCAsmParser::setTargetParser(MCTargetAsmParser &P) {
  assert(!TargetParser && "Target parser is already initialized!");
  TargetParser = &P;
  TargetParser->Initialize(*this);
}

const AsmToken &MCAsmParser::getTok() {
  return getLexer().getTok();
}

bool MCAsmParser::TokError(const Twine &Msg, ArrayRef<SMRange> Ranges) {
  Error(getLexer().getLoc(), Msg, Ranges);
  return true;
}

bool MCAsmParser::ParseExpression(const MCExpr *&Res) {
  SMLoc L;
  return ParseExpression(Res, L);
}

void MCParsedAsmOperand::dump() const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  dbgs() << "  " << *this;
#endif
}
