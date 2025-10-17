/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

//===--- ConstantFolding.h - Utilities for SIL constant folding -*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// This file defines utility functions for constant folding.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SIL_CONSTANTFOLDING_H
#define LANGUAGE_SIL_CONSTANTFOLDING_H

#include "language/SIL/SILInstruction.h"
#include "language/SILOptimizer/Analysis/Analysis.h"
#include "toolchain/ADT/SetVector.h"
#include <functional>

namespace language {

class SILOptFunctionBuilder;

/// Evaluates the constant result of a binary bit-operation.
///
/// The \p ID must be the ID of a binary bit-operation builtin.
APInt constantFoldBitOperation(APInt lhs, APInt rhs, BuiltinValueKind ID);

/// Evaluates the constant result of a floating point comparison.
///
/// The \p ID must be the ID of a floating point builtin operation.
APInt constantFoldComparisonFloat(APFloat lhs, APFloat rhs,
                                  BuiltinValueKind ID);

/// Evaluates the constant result of an integer comparison.
///
/// The \p ID must be the ID of an integer builtin operation.
APInt constantFoldComparisonInt(APInt lhs, APInt rhs, BuiltinValueKind ID);

/// Evaluates the constant result of a binary operation with overflow.
///
/// The \p ID must be the ID of a binary operation with overflow.
APInt constantFoldBinaryWithOverflow(APInt lhs, APInt rhs, bool &Overflow,
                                     toolchain::Intrinsic::ID ID);

/// Evaluates the constant result of a division operation.
///
/// The \p ID must be the ID of a division operation.
APInt constantFoldDiv(APInt lhs, APInt rhs, bool &Overflow, BuiltinValueKind ID);

  /// Evaluates the constant result of an integer cast operation.
  ///
  /// The \p ID must be the ID of a trunc/sext/zext builtin.
APInt constantFoldCast(APInt val, const BuiltinInfo &BI);

/// If `ResultsInError` is not none than errors are diagnosed and
/// `ResultsInError` is set to true in case of an error.
SILValue constantFoldBuiltin(BuiltinInst *BI,
                             std::optional<bool> &ResultsInError);

/// A utility class to do constant folding.
class ConstantFolder {
private:
  SILOptFunctionBuilder &FuncBuilder;

  /// The worklist of the constants that could be folded into their users.
  toolchain::SetVector<SILInstruction *> WorkList;

  /// The assert configuration of SILOptions.
  unsigned AssertConfiguration;

  /// Print diagnostics as part of mandatory constant propagation.
  bool EnableDiagnostics;

  /// Called for each constant folded instruction.
  std::function<void (SILInstruction *)> Callback;

public:
  /// The constructor.
  ///
  /// \param AssertConfiguration The assert configuration of SILOptions.
  /// \param EnableDiagnostics Print diagnostics as part of mandatory constant
  ///                          propagation.
  /// \param Callback Called for each constant folded instruction.
  ConstantFolder(SILOptFunctionBuilder &FuncBuilder,
                 unsigned AssertConfiguration,
                 bool EnableDiagnostics = false,
                 std::function<void (SILInstruction *)> Callback =
                 [](SILInstruction *){}) :
    FuncBuilder(FuncBuilder),
    AssertConfiguration(AssertConfiguration),
    EnableDiagnostics(EnableDiagnostics),
    Callback(Callback) { }

  /// Initialize the worklist with all instructions of the function \p F.
  void initializeWorklist(SILFunction &F);

  /// When asserts are enabled, dumps the worklist for diagnostic
  /// purposes. Without asserts this is a no-op.
  void dumpWorklist() const;

  /// Initialize the worklist with a single instruction \p I.
  void addToWorklist(SILInstruction *I) {
    WorkList.insert(I);
  }

  /// Constant fold everything in the worklist and transitively all uses of
  /// folded instructions.
  SILAnalysis::InvalidationKind processWorkList();
};

} // end namespace language

#endif
