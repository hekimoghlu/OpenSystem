/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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

//===- llvm/TableGen/Error.h - tblgen error handling helpers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains error handling helper routines to pretty-print diagnostic
// messages from tblgen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_ERROR_H
#define LLVM_TABLEGEN_ERROR_H

#include "llvm/Support/SourceMgr.h"

namespace llvm {

class TGError {
  SmallVector<SMLoc, 4> Locs;
  std::string Message;
public:
  TGError(ArrayRef<SMLoc> locs, const std::string &message)
    : Locs(locs.begin(), locs.end()), Message(message) {}

  ArrayRef<SMLoc> getLoc() const { return Locs; }
  const std::string &getMessage() const { return Message; }
};

void PrintWarning(ArrayRef<SMLoc> WarningLoc, const Twine &Msg);
void PrintWarning(const char *Loc, const Twine &Msg);
void PrintWarning(const Twine &Msg);
void PrintWarning(const TGError &Warning);

void PrintError(ArrayRef<SMLoc> ErrorLoc, const Twine &Msg);
void PrintError(const char *Loc, const Twine &Msg);
void PrintError(const Twine &Msg);
void PrintError(const TGError &Error);


extern SourceMgr SrcMgr;


} // end namespace "llvm"

#endif
