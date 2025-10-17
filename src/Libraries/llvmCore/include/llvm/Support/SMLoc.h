/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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

//===- SMLoc.h - Source location for use with diagnostics -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SMLoc class.  This class encapsulates a location in
// source code for use in diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SMLOC_H
#define SUPPORT_SMLOC_H

#include <cassert>

namespace llvm {

/// SMLoc - Represents a location in source code.
class SMLoc {
  const char *Ptr;
public:
  SMLoc() : Ptr(0) {}

  bool isValid() const { return Ptr != 0; }

  bool operator==(const SMLoc &RHS) const { return RHS.Ptr == Ptr; }
  bool operator!=(const SMLoc &RHS) const { return RHS.Ptr != Ptr; }

  const char *getPointer() const { return Ptr; }

  static SMLoc getFromPointer(const char *Ptr) {
    SMLoc L;
    L.Ptr = Ptr;
    return L;
  }
};

/// SMRange - Represents a range in source code.  Note that unlike standard STL
/// ranges, the locations specified are considered to be *inclusive*.  For
/// example, [X,X] *does* include X, it isn't an empty range.
class SMRange {
public:
  SMLoc Start, End;

  SMRange() {}
  SMRange(SMLoc St, SMLoc En) : Start(St), End(En) {
    assert(Start.isValid() == End.isValid() &&
           "Start and end should either both be valid or both be invalid!");
  }
  
  bool isValid() const { return Start.isValid(); }
};
  
} // end namespace llvm

#endif

