/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

//===- TableGenBackend.cpp - Utilities for TableGen Backends ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides useful services for TableGen backends...
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/TableGenBackend.h"
using namespace llvm;

static void printLine(raw_ostream &OS, const Twine &Prefix, char Fill,
                      StringRef Suffix) {
  uint64_t Pos = OS.tell();
  OS << Prefix;
  for (unsigned i = OS.tell() - Pos, e = 80 - Suffix.size(); i != e; ++i)
    OS << Fill;
  OS << Suffix << '\n';
}

void llvm::emitSourceFileHeader(StringRef Desc, raw_ostream &OS) {
  printLine(OS, "/*===- TableGen'erated file ", '-', "*- C++ -*-===*\\");
  printLine(OS, "|*", ' ', "*|");
  printLine(OS, "|* " + Desc, ' ', "*|");
  printLine(OS, "|*", ' ', "*|");
  printLine(OS, "|* Automatically generated file, do not edit!", ' ', "*|");
  printLine(OS, "|*", ' ', "*|");
  printLine(OS, "\\*===", '-', "===*/");
  OS << '\n';
}
