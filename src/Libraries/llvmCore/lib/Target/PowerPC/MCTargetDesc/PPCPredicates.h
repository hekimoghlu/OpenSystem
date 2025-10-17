/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

//===-- PPCPredicates.h - PPC Branch Predicate Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the PowerPC branch predicates.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_POWERPC_PPCPREDICATES_H
#define LLVM_TARGET_POWERPC_PPCPREDICATES_H

namespace llvm {
namespace PPC {
  /// Predicate - These are "(BI << 5) | BO"  for various predicates.
  enum Predicate {
    PRED_ALWAYS = (0 << 5) | 20,
    PRED_LT     = (0 << 5) | 12,
    PRED_LE     = (1 << 5) |  4,
    PRED_EQ     = (2 << 5) | 12,
    PRED_GE     = (0 << 5) |  4,
    PRED_GT     = (1 << 5) | 12,
    PRED_NE     = (2 << 5) |  4,
    PRED_UN     = (3 << 5) | 12,
    PRED_NU     = (3 << 5) |  4
  };
  
  /// Invert the specified predicate.  != -> ==, < -> >=.
  Predicate InvertPredicate(Predicate Opcode);
}
}

#endif
