/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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

//===-- Optimizer/Transforms/MemoryUtils.h ----------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// Author: Tunjay Akbarli
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.toolchain.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility to replace fir.alloca by dynamic allocation and
// deallocation. The exact kind of dynamic allocation is left to be defined by
// the utility user via callbacks (could be fir.allocmem or custom runtime
// calls).
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_COMPABILITY_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H
#define LANGUAGE_COMPABILITY_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H

#include "language/Compability/Optimizer/Dialect/FIROps.h"

namespace mlir {
class RewriterBase;
}

namespace fir {

/// Type of callbacks that indicate if a given fir.alloca must be
/// rewritten.
using MustRewriteCallBack = toolchain::function_ref<bool(fir::AllocaOp)>;

/// Type of callbacks that produce the replacement for a given fir.alloca.
/// It is provided extra information about the dominance of the deallocation
/// points that have been identified, and may refuse to replace the alloca,
/// even if the MustRewriteCallBack previously returned true, in which case
/// it should return a null value.
/// The callback should not delete the alloca, the utility will do it.
using AllocaRewriterCallBack = toolchain::function_ref<mlir::Value(
    mlir::OpBuilder &, fir::AllocaOp, bool allocaDominatesDeallocLocations)>;
/// Type of callbacks that must generate deallocation of storage obtained via
/// AllocaRewriterCallBack calls.
using DeallocCallBack =
    toolchain::function_ref<void(mlir::Location, mlir::OpBuilder &, mlir::Value)>;

/// Utility to replace fir.alloca by dynamic allocations inside \p parentOp.
/// \p MustRewriteCallBack lets the user control which fir.alloca should be
/// replaced. \p AllocaRewriterCallBack lets the user define how the new memory
/// should be allocated. \p DeallocCallBack lets the user decide how the memory
/// should be deallocated. The boolean result indicates if the utility succeeded
/// to replace all fir.alloca as requested by the user. Causes of failures are
/// the presence of unregistered operations, or OpenMP/ACC recipe operations
/// that return memory allocated inside their region.
bool replaceAllocas(mlir::RewriterBase &rewriter, mlir::Operation *parentOp,
                    MustRewriteCallBack, AllocaRewriterCallBack,
                    DeallocCallBack);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MEMORYUTILS_H
