/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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

//===-- llvm/Target/TargetIntrinsicInfo.h - Instruction Info ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the target intrinsic instructions to the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINTRINSICINFO_H
#define LLVM_TARGET_TARGETINTRINSICINFO_H

#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {

class Function;
class Module;
class Type;

//---------------------------------------------------------------------------
///
/// TargetIntrinsicInfo - Interface to description of machine instruction set
///
class TargetIntrinsicInfo {
  TargetIntrinsicInfo(const TargetIntrinsicInfo &) LLVM_DELETED_FUNCTION;
  void operator=(const TargetIntrinsicInfo &) LLVM_DELETED_FUNCTION;
public:
  TargetIntrinsicInfo();
  virtual ~TargetIntrinsicInfo();

  /// Return the name of a target intrinsic, e.g. "llvm.bfin.ssync".
  /// The Tys and numTys parameters are for intrinsics with overloaded types
  /// (e.g., those using iAny or fAny). For a declaration for an overloaded
  /// intrinsic, Tys should point to an array of numTys pointers to Type,
  /// and must provide exactly one type for each overloaded type in the
  /// intrinsic.
  virtual std::string getName(unsigned IID, Type **Tys = 0,
                              unsigned numTys = 0) const = 0;

  /// Look up target intrinsic by name. Return intrinsic ID or 0 for unknown
  /// names.
  virtual unsigned lookupName(const char *Name, unsigned Len) const =0;

  /// Return the target intrinsic ID of a function, or 0.
  virtual unsigned getIntrinsicID(Function *F) const;

  /// Returns true if the intrinsic can be overloaded.
  virtual bool isOverloaded(unsigned IID) const = 0;
  
  /// Create or insert an LLVM Function declaration for an intrinsic,
  /// and return it. The Tys and numTys are for intrinsics with overloaded
  /// types. See above for more information.
  virtual Function *getDeclaration(Module *M, unsigned ID, Type **Tys = 0,
                                   unsigned numTys = 0) const = 0;
};

} // End llvm namespace

#endif
