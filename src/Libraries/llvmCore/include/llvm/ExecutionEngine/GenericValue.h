/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

//===-- GenericValue.h - Represent any type of LLVM value -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The GenericValue class is used to represent an LLVM value of arbitrary type.
//
//===----------------------------------------------------------------------===//


#ifndef GENERIC_VALUE_H
#define GENERIC_VALUE_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

typedef void* PointerTy;
class APInt;

struct GenericValue {
  union {
    double          DoubleVal;
    float           FloatVal;
    PointerTy       PointerVal;
    struct { unsigned int first; unsigned int second; } UIntPairVal;
    unsigned char   Untyped[8];
  };
  APInt IntVal;   // also used for long doubles

  GenericValue() : DoubleVal(0.0), IntVal(1,0) {}
  explicit GenericValue(void *V) : PointerVal(V), IntVal(1,0) { }
};

inline GenericValue PTOGV(void *P) { return GenericValue(P); }
inline void* GVTOP(const GenericValue &GV) { return GV.PointerVal; }

} // End llvm namespace
#endif
