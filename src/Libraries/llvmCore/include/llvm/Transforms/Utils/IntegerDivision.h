/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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

//===- llvm/Transforms/Utils/IntegerDivision.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of 32bit integer division for targets
// that don't have native support. It's largely derived from compiler-rt's
// implementation of __udivsi3, but hand-tuned for targets that prefer less
// control flow.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_UTILS_INTEGERDIVISION_H
#define TRANSFORMS_UTILS_INTEGERDIVISION_H

namespace llvm {
  class BinaryOperator;
}

namespace llvm {

  /// Generate code to calculate the remainder of two integers, replacing Rem
  /// with the generated code. This currently generates code using the udiv
  /// expansion, but future work includes generating more specialized code,
  /// e.g. when more information about the operands are known. Currently only
  /// implements 32bit scalar division (due to udiv's limitation), but future
  /// work is removing this limitation.
  ///
  /// @brief Replace Rem with generated code.
  bool expandRemainder(BinaryOperator *Rem);

  /// Generate code to divide two integers, replacing Div with the generated
  /// code. This currently generates code similarly to compiler-rt's
  /// implementations, but future work includes generating more specialized code
  /// when more information about the operands are known. Currently only
  /// implements 32bit scalar division, but future work is removing this
  /// limitation.
  ///
  /// @brief Replace Div with generated code.
  bool expandDivision(BinaryOperator* Div);

} // End llvm namespace

#endif
