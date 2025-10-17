/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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

//===- AsmCond.h - Assembly file conditional assembly  ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ASMCOND_H
#define ASMCOND_H

namespace llvm {

/// AsmCond - Class to support conditional assembly
///
/// The conditional assembly feature (.if, .else, .elseif and .endif) is
/// implemented with AsmCond that tells us what we are in the middle of 
/// processing.  Ignore can be either true or false.  When true we are ignoring
/// the block of code in the middle of a conditional.

class AsmCond {
public:
  enum ConditionalAssemblyType {
    NoCond,     // no conditional is being processed
    IfCond,     // inside if conditional
    ElseIfCond, // inside elseif conditional
    ElseCond    // inside else conditional
  };

  ConditionalAssemblyType TheCond;
  bool CondMet;
  bool Ignore;

  AsmCond() : TheCond(NoCond), CondMet(false), Ignore(false) {}
};

} // end namespace llvm

#endif
