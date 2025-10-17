/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

//===--- ProgramTerminationAnalysis.h ---------------------------*- C++ -*-===//
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
///
/// \file
///
/// This is an analysis which determines if a block is a "program terminating
/// block". Define a program terminating block is defined as follows:
///
/// 1. A block at whose end point according to the SIL model, the program must
/// end. An example of such a block is one that includes a call to fatalError.
/// 2. Any block that is joint post-dominated by program terminating blocks.
///
/// For now we only identify instances of 1. But the analysis could be extended
/// appropriately via simple dataflow or through the use of post-dominator
/// trees.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SILOPTIMIZER_ANALYSIS_PROGRAMTERMINATIONANALYSIS_H
#define LANGUAGE_SILOPTIMIZER_ANALYSIS_PROGRAMTERMINATIONANALYSIS_H

#include "language/SILOptimizer/Analysis/ARCAnalysis.h"
#include "toolchain/ADT/SmallPtrSet.h"

namespace language {

class ProgramTerminationFunctionInfo {
  toolchain::SmallPtrSet<const SILBasicBlock *, 4> ProgramTerminatingBlocks;

public:
  ProgramTerminationFunctionInfo(const SILFunction *F) {
    for (const auto &BB : *F) {
      if (!isARCInertTrapBB(&BB))
        continue;
      ProgramTerminatingBlocks.insert(&BB);
    }
  }

  bool isProgramTerminatingBlock(const SILBasicBlock *BB) const {
    return ProgramTerminatingBlocks.count(BB);
  }
};

} // end language namespace

#endif
