/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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

//===--- Passes.h - Codira Compiler SIL Pass Entrypoints ---------*- C++ -*-===//
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
//  This file declares the main entrypoints to SIL passes.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SILOPTIMIZER_PASSMANAGER_PASSES_H
#define LANGUAGE_SILOPTIMIZER_PASSMANAGER_PASSES_H

#include "language/SIL/SILModule.h"

namespace language {
  class SILOptions;
  class SILTransform;
  class SILModuleTransform;

  namespace irgen {
    class IRGenModule;
  }

  /// Run all the SIL diagnostic passes on \p M.
  ///
  /// \returns true if the diagnostic passes produced an error
  bool runSILDiagnosticPasses(SILModule &M);

  /// Run all the SIL performance optimization passes on \p M.
  void runSILOptimizationPasses(SILModule &M);

  /// Run all SIL passes for -Onone on module \p M.
  void runSILPassesForOnone(SILModule &M);

  /// Run the SIL lower hop-to-actor pass on \p M.
  bool runSILLowerHopToActorPass(SILModule &M);

/// Run the SIL ownership eliminator pass on \p M.
  bool runSILOwnershipEliminatorPass(SILModule &M);

  void runSILOptimizationPassesWithFileSpecification(SILModule &Module,
                                                     StringRef FileName);

  /// Detect and remove unreachable code. Diagnose provably unreachable
  /// user code.
  void performSILDiagnoseUnreachable(SILModule *M);

  /// Remove dead functions from \p M.
  void performSILDeadFunctionElimination(SILModule *M);

  /// Convert SIL to a lowered form suitable for IRGen.
  void runSILLoweringPasses(SILModule &M);

  /// Perform SIL Inst Count on M if needed.
  void performSILInstCountIfNeeded(SILModule *M);

  /// Identifiers for all passes. Used to procedurally create passes from
  /// lists of passes.
  enum class PassKind {
#define PASS(ID, TAG, NAME) ID,
#include "Passes.def"
    invalidPassKind,
    numPasses = invalidPassKind,
  };

  PassKind PassKindFromString(StringRef ID);
  StringRef PassKindID(PassKind Kind);
  StringRef PassKindTag(PassKind Kind);

#define PASS(ID, TAG, NAME) \
  SILTransform *create##ID();
#define IRGEN_PASS(ID, TAG, NAME)
#include "Passes.def"

} // end namespace language

#endif
